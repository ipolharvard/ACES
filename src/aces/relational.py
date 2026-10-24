"""Identity-preserving, YAML-defined event queries for MEDS cohorts.

Unlike count windows, these queries retain event columns until linked events have been selected. Expressions
are parsed into Polars expressions; Python is never executed. The original window-query engine remains
available unchanged.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import ruamel.yaml

from .config import PlainPredicateConfig
from .utils import parse_timedelta


def expression(source: str, columns: set[str]) -> pl.Expr:
    """Compile a restricted expression language, checking every column reference."""
    binary = {ast.Add: operator.add, ast.Sub: operator.sub}
    comparisons = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    def visit(node):
        if isinstance(node, ast.Constant):
            return pl.lit(node.value)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name = f"{node.value.id}__{node.attr}"
            if name not in columns:
                raise ValueError(f"Unknown event field: {node.value.id}.{node.attr}")
            return pl.col(name)
        if isinstance(node, ast.BoolOp):
            values = [visit(value) for value in node.values]
            return pl.all_horizontal(values) if isinstance(node.op, ast.And) else pl.any_horizontal(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ~visit(node.operand)
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in (int, float)
        ):
            value = node.operand.value
            return pl.lit(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.Compare):
            parts = []
            left = node.left
            for op, right in zip(node.ops, node.comparators, strict=True):
                if type(op) not in comparisons:
                    raise ValueError("Use is_null()/is_not_null() for null checks")
                parts.append(comparisons[type(op)](visit(left), visit(right)))
                left = right
            return pl.all_horizontal(parts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name == "duration" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                value = node.args[0].value
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("duration() requires a nonempty duration string")
                return pl.lit(parse_timedelta(value))
            args = [visit(arg) for arg in node.args]
            if name == "is_null" and len(args) == 1:
                return args[0].is_null()
            if name == "is_not_null" and len(args) == 1:
                return args[0].is_not_null()
            if name == "coalesce" and args:
                return pl.coalesce(args)
            if name == "min_time" and args:
                return pl.min_horizontal(args)
            if name == "max_time" and args:
                return pl.max_horizontal(args)
        raise ValueError(f"Unsupported expression syntax: {ast.dump(node)}")

    if not isinstance(source, str):
        raise ValueError("Expressions must be strings")
    try:
        return visit(ast.parse(source, mode="eval").body)
    except SyntaxError as error:
        raise ValueError(f"Invalid expression: {source!r}") from error


def column_condition(column: str, condition: Any) -> pl.Expr:
    """Literal equality plus explicit, null-safe column operators."""
    value = pl.col(column)
    if condition is None:
        return value.is_null()
    if not isinstance(condition, dict):
        return value == condition
    if len(condition) != 1:
        raise ValueError(f"Expected one operator for {column!r}")
    op, target = next(iter(condition.items()))
    if op in {"is_null", "is_not_null"}:
        if target is not True:
            raise ValueError(f"{op} requires true")
        return value.is_null() if op == "is_null" else value.is_not_null()
    operators = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }
    if op in operators and target is not None:
        return operators[op](value, target)
    if op == "in" and isinstance(target, list):
        return value.is_in(target)
    raise ValueError(f"Unsupported column operator: {op!r}")


@dataclass
class RelationalTaskConfig:
    """Opt-in event-query configuration; shared definitions can be supplied separately."""

    definitions: dict
    task: dict
    label_window: str = "label"
    index_timestamp_window: str = "index_timestamp"

    @classmethod
    def from_dict(cls, data: dict, predicates_path: Path | str | None = None):
        data = dict(data)
        if data.pop("engine", None) != "relational" or data.pop("schema_version", None) != 1:
            raise ValueError("Relational queries require engine: relational and schema_version: 1")
        shared = {}
        if predicates_path:
            shared = ruamel.yaml.YAML(typ="safe", pure=True).load(Path(predicates_path).read_text())
        definitions = {}
        for key in ("predicates", "identifiers", "event_sets"):
            definitions[key] = {**shared.pop(key, {}), **data.pop(key, {})}
        shared.pop("description", None)
        shared.pop("metadata", None)
        if shared:
            raise ValueError(f"Unknown shared definition keys: {sorted(shared)}")
        data.pop("description", None)
        data.pop("metadata", None)
        allowed = {"trigger", "relations", "include", "index_timestamp", "label", "output", "distinct_by"}
        if set(data) - allowed:
            raise ValueError(f"Unknown relational task keys: {sorted(set(data) - allowed)}")
        for key in ("trigger", "index_timestamp", "label"):
            if key not in data:
                raise ValueError("Task requires a label and an index_timestamp, and a trigger")
        return cls(definitions, data)


def extract(cfg: RelationalTaskConfig, source: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Extract labels with episode identity, temporal links, and explicit censoring.

    Matching is always subject-local and null identifiers never match. Each relation selects one first/last
    event, leaving missing links null. All temporal and eligibility policies are expressions in the
    configuration.
    """
    source = source.lazy()
    schema = source.collect_schema()
    required = {"subject_id", "time", "code"}
    if missing := required - set(schema.names()):
        raise ValueError(f"Missing MEDS columns: {sorted(missing)}")
    if not isinstance(schema["time"], pl.Datetime):
        raise ValueError("MEDS time must be a Datetime column")
    if schema["time"].time_zone is not None:
        source = source.with_columns(pl.col("time").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
    # Project before materializing histories. Large MEDS payload columns that
    # neither predicates nor queries consume must not enter the event tables.
    fields = {"subject_id", "time", "code"}
    for raw in cfg.definitions["identifiers"].values():
        fields.update(raw if isinstance(raw, list) else [raw])
    for definition in cfg.definitions["predicates"].values():
        if "value_min" in definition or "value_max" in definition:
            fields.add("numeric_value")
        fields.update(definition.get("other_cols", {}))
        fields.update(set(definition) - set(PlainPredicateConfig.__dataclass_fields__) - {"expr"})
    for spec in [cfg.task, *cfg.definitions["event_sets"].values()]:
        identity_roles = [
            *spec.get("distinct_by", []),
            *[role for link in spec.get("relations", {}).values() for role in link.get("match", [])],
        ]
        for role in identity_roles:
            raw = cfg.definitions["identifiers"].get(role, role)
            fields.update(raw if isinstance(raw, list) else [raw])
        expressions = [
            spec.get("include", "True"),
            spec.get("label", "True"),
            spec.get("index_timestamp", "trigger.time"),
            *spec.get("output", {}).values(),
            *[link.get("where", "True") for link in spec.get("relations", {}).values()],
        ]
        for value in expressions:
            for node in ast.walk(ast.parse(value, mode="eval")):
                if isinstance(node, ast.Attribute):
                    raw = cfg.definitions["identifiers"].get(node.attr, node.attr)
                    fields.update(raw if isinstance(raw, list) else [raw])
    source = (
        source.with_row_index("_event_id")
        .select("_event_id", *sorted(fields & set(schema.names())))
        .with_columns(pl.col("time").cast(pl.Datetime("us")))
    )
    # Collect once after projection/filtering. Subsequent queries retain a lazy plan.
    events = source.filter(pl.col("time").is_not_null()).collect().lazy()
    names = events.collect_schema().names()
    roles = cfg.definitions["identifiers"]
    cache = {}
    building = set()

    def role_columns(role):
        columns = roles.get(role, role)
        columns = columns if isinstance(columns, list) else [columns]
        if not columns or any(column not in names for column in columns):
            raise ValueError(f"Missing identifier columns for {role!r}: {columns}")
        return columns

    def alias(frame, prefix):
        fields = frame.collect_schema().names()
        result = frame.select(pl.col(field).alias(f"{prefix}__{field}") for field in fields)
        for role, raw in roles.items():
            if isinstance(raw, str) and raw in fields and role not in fields:
                result = result.with_columns(pl.col(f"{prefix}__{raw}").alias(f"{prefix}__{role}"))
        return result

    def predicate(name):
        if name == "_ANY_EVENT":
            return pl.lit(True)
        definition = cfg.definitions["predicates"].get(name)
        if definition is None:
            raise ValueError(f"Unknown predicate: {name!r}")
        if "expr" in definition:
            # Preserve the existing ACES derived-predicate syntax.
            from .config import DerivedPredicateConfig

            derived = DerivedPredicateConfig(**definition)
            children = [predicate(child) for child in derived.input_predicates]
            return pl.all_horizontal(children) if derived.is_and else pl.any_horizontal(children)
        plain_fields = set(PlainPredicateConfig.__dataclass_fields__) - {"other_cols"}
        plain = {key: value for key, value in definition.items() if key in plain_fields}
        other = {
            **definition.get("other_cols", {}),
            **{
                key: value
                for key, value in definition.items()
                if key not in plain_fields and key != "other_cols"
            },
        }
        return pl.all_horizontal(
            PlainPredicateConfig(**plain).MEDS_eval_expr(),
            *[column_condition(key, value) for key, value in other.items()],
        ).fill_null(False)

    def event_set(name):
        if name in cache:
            return cache[name]
        if name in building:
            raise ValueError(f"Cyclic event-set dependency: {name}")
        building.add(name)
        if name in cfg.definitions["event_sets"]:
            spec = cfg.definitions["event_sets"][name]
            selected = query(spec)
            result = selected.select(pl.col(f"trigger__{field}").alias(field) for field in names)
        else:
            result = events.filter(predicate(name))
        building.remove(name)
        # Reused event sets are small clinical boundary tables, not the full history.
        cache[name] = result.collect().lazy()
        return cache[name]

    def query(spec):
        if unknown := set(spec) - {
            "trigger",
            "relations",
            "include",
            "index_timestamp",
            "label",
            "output",
            "distinct_by",
        }:
            raise ValueError(f"Unknown query keys: {sorted(unknown)}")
        triggers = event_set(spec["trigger"])
        if spec.get("distinct_by"):
            keys = ["subject_id", "time"]
            for role in spec["distinct_by"]:
                keys.extend(role_columns(role))
            triggers = triggers.unique(subset=list(dict.fromkeys(keys)), maintain_order=True)
        anchor = alias(triggers, "trigger").with_row_index("_sample_id")
        anchor = anchor.with_columns(pl.col("trigger__subject_id").alias("subject_id"))
        for relation, link in spec.get("relations", {}).items():
            if not relation.isidentifier() or relation in {"trigger", "event"}:
                raise ValueError(f"Invalid relation name: {relation}")
            if unknown := set(link) - {"predicate", "match", "where", "select"}:
                raise ValueError(f"Unknown relation keys: {sorted(unknown)}")
            selection = link.get("select", "first")
            if selection not in {"first", "last"}:
                raise ValueError("Relation select must be first or last")
            candidates = alias(event_set(link["predicate"]), "event")
            left_keys = ["subject_id"]
            right_keys = ["event__subject_id"]
            for role in link.get("match", []):
                for field in role_columns(role):
                    if f"event__{field}" not in right_keys:
                        left_keys.append(f"trigger__{field}")
                        right_keys.append(f"event__{field}")
            # Aggregate anchor-independent relations before joining, avoiding a
            # history x anchor expansion for observation-end and other global links.
            condition = link.get("where", "True")
            references = {
                node.value.id
                for node in ast.walk(ast.parse(condition, mode="eval"))
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            }
            if references <= {"event"}:
                selected = candidates.filter(expression(condition, set(candidates.collect_schema().names())))
                selected = selected.sort(["event__time", "event___event_id"])
                selected = selected.group_by(right_keys).agg(getattr(pl.all(), selection)())
                right_subject = "event__subject_id" not in right_keys[1:]
                if right_subject:
                    selected = selected.with_columns(pl.col("event__subject_id").alias("_join_subject"))
                    right_keys = ["_join_subject", *right_keys[1:]]
                projected = selected.rename(
                    {
                        field: field.replace("event__", f"{relation}__", 1)
                        for field in selected.collect_schema().names()
                        if field.startswith("event__")
                    }
                )
                right_keys = [key.replace("event__", f"{relation}__", 1) for key in right_keys]
                anchor = anchor.join(
                    projected, left_on=left_keys, right_on=right_keys, how="left", coalesce=False
                ).drop("_join_subject", strict=False)
            else:
                selected = anchor.join(
                    candidates, left_on=left_keys, right_on=right_keys, how="inner", coalesce=False
                )
                selected = selected.filter(expression(condition, set(selected.collect_schema().names())))
                selected = selected.sort(["_sample_id", "event__time", "event___event_id"])
                fields = candidates.collect_schema().names()
                selected = selected.group_by("_sample_id").agg(
                    getattr(pl.col(field), selection)().alias(field.replace("event__", f"{relation}__", 1))
                    for field in fields
                )
                anchor = anchor.join(selected, on="_sample_id", how="left")
        if "include" in spec:
            anchor = anchor.filter(
                expression(spec["include"], set(anchor.collect_schema().names())).fill_null(False)
            )
        return anchor

    result = query(cfg.task)
    fields = set(result.collect_schema().names())
    output = {
        "subject_id": pl.col("subject_id").cast(pl.Int64),
        "index_timestamp": expression(cfg.task["index_timestamp"], fields).cast(pl.Datetime("us")),
        "label": expression(cfg.task["label"], fields).fill_null(False).cast(pl.Boolean),
        "trigger_event_id": pl.col("trigger___event_id").cast(pl.Int64),
        "trigger_time": pl.col("trigger__time"),
        "trigger_data": pl.struct(pl.col(f"trigger__{field}").alias(field) for field in names),
    }
    output.update({key: expression(value, fields) for key, value in cfg.task.get("output", {}).items()})
    if set(cfg.task.get("output", {})) & {
        "subject_id",
        "index_timestamp",
        "label",
        "trigger_event_id",
        "trigger_time",
        "trigger_data",
    }:
        raise ValueError("Output cannot replace reserved label columns")
    result = (
        result.select(value.alias(key) for key, value in output.items())
        .sort(["subject_id", "index_timestamp", "trigger_event_id"])
        .collect()
    )
    if result["index_timestamp"].null_count():
        raise ValueError("Prediction timestamps must not be null")
    return result

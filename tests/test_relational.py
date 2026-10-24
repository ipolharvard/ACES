"""Identity-preserving event queries and backward-compatible column predicates."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from hydra import compose, initialize_config_dir

from aces.config import PlainPredicateConfig, TaskExtractorConfig
from aces.relational import RelationalTaskConfig, expression, extract
from aces.run import config_yaml, main

START = datetime(2020, 1, 1)


def frame(rows):
    return pl.DataFrame(
        {
            "subject_id": [row[0] for row in rows],
            "time": [START + timedelta(hours=row[1]) for row in rows],
            "code": [row[2] for row in rows],
            "hadm_id": pl.Series([row[3] for row in rows], dtype=pl.Int64),
            "stay_id": pl.Series([row[4] for row in rows], dtype=pl.Int64),
        }
    )


def config(**task):
    return RelationalTaskConfig.from_dict(
        {
            "engine": "relational",
            "schema_version": 1,
            "predicates": {
                "ed": {"code": "ED"},
                "adm": {"code": "ADM"},
                "dc": {"code": "DC"},
                "death": {"code": "DEATH"},
            },
            "identifiers": {"hospitalization": "hadm_id", "ed_visit": "stay_id"},
            "trigger": "ed",
            "index_timestamp": "trigger.time",
            "label": "is_not_null(trigger.hadm_id)",
            **task,
        }
    )


def test_null_predicates_are_explicit_and_legacy_null_literal_is_fixed():
    df = pl.DataFrame({"code": ["ED", "ED", "OTHER"], "hadm_id": [None, 2**63 - 1, None]})
    assert df.select(
        PlainPredicateConfig("ED", other_cols={"hadm_id": None}).MEDS_eval_expr()
    ).to_series().to_list() == [True, False, False]
    assert df.select(
        PlainPredicateConfig("ED", other_cols={"hadm_id": {"is_not_null": True}}).MEDS_eval_expr()
    ).to_series().to_list() == [False, True, False]


def test_subject_and_hospital_id_linking_does_not_match_next_unrelated_admission():
    data = frame(
        [
            (1, 0, "ED", 101, 501),
            (1, 1, "ADM", 102, None),
            (1, 2, "ADM", 101, None),
            (2, 0.5, "ADM", 101, None),
        ]
    )
    cfg = config(
        relations={
            "admission": {
                "predicate": "adm",
                "match": ["hospitalization"],
                "where": "event.time > trigger.time",
            }
        },
        label='admission.time == trigger.time + duration("2h")',
    )
    assert extract(cfg, data)["label"].to_list() == [True]


def test_null_identifiers_never_link_even_with_same_subject():
    data = frame([(1, 0, "ED", None, 501), (1, 1, "ADM", None, None)])
    cfg = config(
        relations={"admission": {"predicate": "adm", "match": ["hospitalization"]}},
        label="is_not_null(admission.time)",
    )
    assert extract(cfg, data)["label"].to_list() == [False]


def test_same_timestamp_different_encounters_keep_distinct_triggers():
    data = frame([(1, 0, "ED", None, 501), (1, 0, "ED", 101, 502)])
    result = extract(config(), data)
    assert result["label"].to_list() == [False, True]
    assert result["trigger_event_id"].n_unique() == 2


def test_missing_link_stays_null_and_first_and_last_have_defined_order():
    data = frame(
        [(1, 0, "ED", 101, 501), (1, 1, "ADM", 101, None), (1, 2, "ADM", 101, None), (2, 0, "ED", 201, 601)]
    )
    cfg = config(
        relations={
            "first": {"predicate": "adm", "match": ["hospitalization"], "where": "event.time > trigger.time"},
            "last": {"predicate": "adm", "match": ["hospitalization"], "select": "last"},
        },
        label="last.time - first.time == duration('1h')",
    )
    assert extract(cfg, data)["label"].to_list() == [True, False]


def test_early_positive_and_negative_followup_are_separate_eligibility_paths():
    data = frame(
        [
            (1, 0, "ED", None, 501),
            (1, 1, "ADM", 101, None),
            (2, 0, "ED", None, 601),
            (2, 2, "OBS", None, None),
            (3, 0, "ED", None, 701),
            (3, 72, "OBS", None, None),
        ]
    )
    cfg = config(
        relations={
            "admission": {"predicate": "adm", "where": "event.time > trigger.time"},
            "end": {"predicate": "_ANY_EVENT", "select": "last"},
        },
        include='is_not_null(admission.time) or end.time >= trigger.time + duration("72h")',
        label='is_not_null(admission.time) and admission.time <= trigger.time + duration("72h")',
    )
    result = extract(cfg, data)
    assert result["subject_id"].to_list() == [1, 3]
    assert result["label"].to_list() == [True, False]


def test_event_sets_can_pair_composite_identity_and_filter_zero_duration():
    data = frame(
        [(1, 0, "ADM", 101, 501), (1, 0, "DC", 101, 501), (1, 1, "ADM", 101, 502), (1, 2, "DC", 101, 502)]
    )
    cfg = config(trigger="valid", label="True")
    cfg.definitions["identifiers"]["episode"] = ["hadm_id", "stay_id"]
    cfg.definitions["event_sets"]["valid"] = {
        "trigger": "adm",
        "relations": {"dc": {"predicate": "dc", "match": ["episode"], "where": "event.time >= trigger.time"}},
        "include": "dc.time > trigger.time",
    }
    result = extract(cfg, data)
    assert result["index_timestamp"].to_list() == [START + timedelta(hours=1)]


def test_empty_cohort_has_typed_label_schema():
    result = extract(config(include="False"), frame([(1, 0, "ED", None, 501)]))
    assert result.is_empty()
    assert result.schema["label"] == pl.Boolean
    assert result.schema["index_timestamp"] == pl.Datetime("us")


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('true')",
        "event.time.year",
        "unknown.time",
        "[x for x in []]",
        "trigger.time is None",
    ],
)
def test_expression_language_rejects_python_execution_and_unknown_fields(source):
    with pytest.raises(ValueError):
        expression(source, {"trigger__time"})


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        ("-1", [True, True, False, False, False, False]),
        ("+1", [True, True, True, False, False, False]),
        ("-1.5", [True, False, False, False, False, False]),
        ("+1.5", [True, True, True, True, False, False]),
    ],
)
def test_signed_numeric_thresholds(threshold, expected):
    data = frame([(i, 0, "ED", 101, 501) for i in range(6)]).with_columns(
        pl.Series("numeric_value", [-2.0, -1.25, 0.0, 1.25, 2.0, None])
    )
    cfg = config(label=f"trigger.numeric_value < {threshold}")
    assert extract(cfg, data)["label"].to_list() == expected


def test_load_dispatch_preserves_native_window_configs(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(
        "engine: relational\nschema_version: 1\ntrigger: ed\nlabel: 'True'\nindex_timestamp: trigger.time\n"
    )
    assert isinstance(TaskExtractorConfig.load(path), RelationalTaskConfig)
    path.write_text("predicates:\n  ed:\n    code: ED\ntrigger: ed\nwindows: {}\n")
    assert isinstance(TaskExtractorConfig.load(path), TaskExtractorConfig)


def test_empty_task_still_validates_missing_predicates_path(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text("")
    with pytest.raises(FileNotFoundError, match="Cannot load missing predicates file"):
        TaskExtractorConfig.load(path, tmp_path / "missing.yaml")


@pytest.mark.parametrize("nested", [False, True], ids=["task", "event_set"])
@pytest.mark.parametrize("operation", ["match", "distinct_by"])
def test_physical_identity_fields_are_projected(operation, nested):
    data = frame([(1, 0, "ED", 101, 501), (1, 0, "ED", 102, 502), (1, 1, "ADM", 102, 502)])
    cfg = config(label="True")
    cfg.definitions["identifiers"] = {}
    spec = {"trigger": "ed"}
    if operation == "match":
        spec.update(
            relations={"admission": {"predicate": "adm", "match": ["hadm_id"]}},
            include="is_not_null(admission.time)",
        )
    else:
        spec["distinct_by"] = ["hadm_id"]
    if nested:
        cfg.definitions["event_sets"]["selected"] = spec
        cfg.task["trigger"] = "selected"
    else:
        cfg.task.update(spec)
    result = extract(cfg, data)
    assert result["trigger_event_id"].to_list() == ([1] if operation == "match" else [0, 1])


@pytest.mark.parametrize("condition", ["True", "event.time > trigger.time"])
@pytest.mark.parametrize("subject_in_episode", [False, True])
def test_overlapping_composite_match_keys(condition, subject_in_episode):
    data = frame(
        [
            (1, 0, "ED", 101, 501),
            (1, 1, "ADM", 101, 502),
            (2, 1, "ADM", 101, 501),
            (1, 2, "ADM", 101, 501),
        ]
    )
    cfg = config(
        relations={
            "admission": {
                "predicate": "adm",
                "match": ["hospitalization", "episode"],
                "where": condition,
            }
        },
        label="admission.time == trigger.time + duration('2h')",
    )
    cfg.definitions["identifiers"]["episode"] = (
        ["subject_id", "hadm_id", "stay_id"] if subject_in_episode else ["hadm_id", "stay_id"]
    )
    assert extract(cfg, data)["label"].to_list() == [True]


@pytest.mark.parametrize("reference", ["trigger", "admission"])
@pytest.mark.parametrize("field", ["{ref}.time", "{ref} .time", "({ref}).time"])
def test_relation_dependencies_accept_equivalent_expression_spacing(reference, field):
    data = frame(
        [(1, 0, "DC", 101, 501), (1, 1, "ED", 101, 501), (1, 2, "ADM", 101, 501), (1, 3, "DC", 101, 501)]
    )
    cfg = config(
        relations={
            "admission": {"predicate": "adm"},
            "discharge": {"predicate": "dc", "where": f"event.time > {field.format(ref=reference)}"},
        },
        label="discharge.time == trigger.time + duration('2h')",
    )
    assert extract(cfg, data)["label"].to_list() == [True]


@pytest.mark.parametrize("shared_directory", [False, True])
def test_relational_run_creates_audit_output_directory(tmp_path, shared_directory):
    data_path = tmp_path / "events.parquet"
    frame([(1, 0, "ED", 101, 501)]).write_parquet(data_path)
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        "engine: relational\nschema_version: 1\ntrigger: _ANY_EVENT\n"
        "label: 'True'\nindex_timestamp: trigger.time\n"
        "output:\n  prediction_time: trigger.time + duration('1h')\n  boolean_value: 'False'\n"
    )
    audit_path = tmp_path / "audit" / "nested" / "audit.parquet"
    output_path = (audit_path.parent if shared_directory else tmp_path / "labels") / "labels.parquet"
    with initialize_config_dir(version_base=None, config_dir=str(config_yaml.parent.resolve())):
        assert compose(config_name=config_yaml.stem).audit_filepath is None
        cfg = compose(
            config_name=config_yaml.stem,
            overrides=[
                f"config_path={config_path}",
                "data.standard=meds",
                f"data.path={data_path}",
                f"audit_filepath={audit_path}",
                f"output_filepath={output_path}",
            ],
        )
    main.__wrapped__(cfg)
    audit = pl.read_parquet(audit_path)
    labels = pl.read_parquet(output_path)
    assert audit["label"].to_list() == labels["boolean_value"].to_list() == [True]
    assert "trigger_data" in audit.columns
    assert "trigger_data" not in labels.columns
    assert audit["prediction_time"].to_list() == [START + timedelta(hours=1)]
    assert labels["prediction_time"].to_list() == [START]
    assert audit["boolean_value"].to_list() == [False]

# Relational event queries

Use `engine: relational` and `schema_version: 1` for MEDS tasks that need event identity,
nullable metadata, or comparisons between linked events. The original count-window engine
remains the default for existing configurations. No additional dependencies are required.

```yaml
engine: relational
schema_version: 1
identifiers:
  hospitalization: hadm_id
predicates:
  ed_registration:
    code: ED//ADMISSION
  hospital_admission:
    code: HOSPITAL//ADMISSION
trigger: ed_registration
relations:
  admission:
    predicate: hospital_admission
    match: [hospitalization]
    select: first
include: >-
  is_null(trigger.hadm_id) or
  (is_not_null(admission.time) and admission.time > trigger.time)
index_timestamp: trigger.time
label: is_not_null(trigger.hadm_id)
output:
  linked_admission_time: admission.time
```

Run this with the existing `aces-cli config_path=... data.standard=meds data.path=...`
interface. The Python API is `aces.relational.extract(config, lazy_meds_frame)`, after loading
the task with `TaskExtractorConfig.load`. The MEDS CLI writes `subject_id`, `prediction_time`,
and `boolean_value`. Set `audit_filepath` to preserve linked fields before label-schema
alignment. Input shards must contain complete subject histories within a fold.

`identifiers` maps logical roles to physical columns, or lists of columns for composite
keys. `match` joins those keys and subject identity against the trigger. Null identifiers do
not match. `where` filters candidate events and may reference `event`, `trigger`, and previously
defined relations. Relations are evaluated in YAML order and select the first or last event
by time, with source row order breaking timestamp ties. Missing relations retain null fields.
`distinct_by` optionally identifies trigger episodes to deduplicate using subject, timestamp,
and the listed identifier roles. Different identities at one timestamp remain distinct.

Expressions support qualified event fields, constants, comparisons, `and`, `or`, `not`,
time addition/subtraction, `duration("30d")`, `is_null`, `is_not_null`, `coalesce`, `min_time`,
and `max_time`. Functions and field references are validated; expressions compile to Polars
rather than executing Python. Null eligibility is false, and null labels become false.
Prediction timestamps must be non-null. Specify endpoint inclusivity explicitly through
comparisons such as `event.time > trigger.time` and `event.time <= trigger.time + duration("12h")`.

`event_sets` define reusable filtered event streams with their own trigger, relations, and
inclusion expression. For example, pair admission and discharge using a stay identifier,
then retain only `discharge.time > trigger.time` before using those admissions as outcomes.
Dependency cycles are errors. Shared `predicates`, `identifiers`, and `event_sets` can be
loaded with `predicates_path`; task-local definitions override shared definitions.

Plain predicates retain the original code/numeric-value syntax and `and(...)`/`or(...)`
derived predicates. Additional-column conditions accept literal equality, `null` for a null
check, or explicit operators such as `{is_not_null: true}`, `{ge: 5}`, and `{in: [a, b]}`.
Both flattened predicate fields and the nested `other_cols` mapping are accepted. Null-safe
column conditions are also available to the original window-query engine.

The relational engine currently supports dynamic MEDS events. Use the original engine for
ESGPT, direct predicate tables, and static demographic constraints. Timestamps normalize to
UTC-naive microseconds when timezone-aware; naive source times remain naive. Event timestamps
are used as supplied: delayed availability and date-only precision must be addressed in the
source representation or task definitions.

# Dataset Versions

A `Dataset` is a stable identity: a name, an id and some metadata. Adding or
replacing examples changes what that dataset contains, which means a run
recorded last month cannot be reproduced and two runs cannot be compared with
any confidence that they saw the same test set.

A `DatasetVersion` is an immutable snapshot of a dataset's examples. Publishing
one freezes the exact set of examples a run reads, so an evaluation stays
reproducible and membership changes between test sets stay visible.

## Publishing a version

```python
from trulens.core import TruSession

session = TruSession()

version = session.create_dataset_version(
    dataset_name="support-quality",
    dataframe=examples,
    column_spec={
        "input": "question",
        "ground_truth_output": "expected_answer",
        "metadata": "metadata",
    },
    splits={"regression": ["case-1", "case-2"]},
    description="Reviewed August failures",
)

print(version.dataset_version_id)
```

The `column_spec` maps dataset version item fields to columns in your
dataframe. The canonical keys are `input`, `input_id`, `expected_response`,
`expected_contexts` and `metadata`; the legacy `GroundTruth` spellings
(`query`, `query_id`, `expected_chunks`, `meta`) and the reserved
`RunConfig.dataset_spec` keys (`ground_truth_output`, `record_root.input`) are
accepted too, so the same mapping can be handed to both APIs. Keys that do not
name an item field are ignored.

Examples can also be published from a sequence of
[GroundTruth][trulens.core.schema.groundtruth.GroundTruth] entries:

```python
version = session.create_dataset_version(
    dataset_name="support-quality",
    ground_truths=ground_truths,
)
```

## Content addressing and idempotency

A version id is a hash of the dataset, the ordered contents of the version and
its source metadata. Two consequences follow:

- **Publishing identical content is idempotent.** The second call returns the
  version that already exists rather than creating a duplicate.
- **Published versions are never updated.** `description` and
  `parent_dataset_version_id` are provenance annotations that are deliberately
  excluded from the hash, so re-publishing the same examples under a new
  description returns the original version, description and all.

Item ids are content-addressed too, from the normalized example content
(`input`, `expected_response`, `expected_contexts`) and the optional
caller-supplied `input_id`. An item's `metadata` and `splits` are properties of
the item *within a version* and do not take part in its identity, so
re-annotating an example produces a new version without making membership
comparison report the example as removed and re-added.

## Loading a version

Pass only a name to resolve the latest version, or pin an exact snapshot:

```python
latest = session.get_dataset_version(dataset_name="support-quality")

pinned = session.get_dataset_version(
    dataset_version_id=version.dataset_version_id,
)

session.list_dataset_versions("support-quality")  # oldest first
```

Named splits are carried on the loaded items:

```python
regression_cases = pinned.split("regression")
```

## Comparing versions

```python
diff = session.compare_dataset_versions(
    baseline.dataset_version_id,
    candidate.dataset_version_id,
)

print(len(diff.added), len(diff.removed), len(diff.unchanged))
```

## Pinning a version to a run

A run can record which snapshot it read, and read that snapshot back:

```python
from trulens.core.run import RunConfig

run = app.add_run(
    RunConfig(
        run_name="august-regression",
        dataset_name="support-quality",
        dataset_spec={"input": "question"},
        dataset_version_id=version.dataset_version_id,
    )
)

run.start()  # loads the pinned snapshot instead of the live source
```

The version id is persisted in the run's source info, shows up in
`run.describe()`, and is reported by `RunDiff.provenance()` when two runs are
compared. Comparing two runs pinned to *different* versions logs a warning,
since the metric deltas then mix changes in the app with changes in the test
set.

## Existing datasets

Datasets whose examples predate versioning are exposed as **version zero**,
reconstructed from their `GroundTruth` rows. The original ground truth payloads
are read but never rewritten. Version zero is materialized as a real row the
first time a newer version is published for that dataset, so it stays loadable
afterwards and becomes the parent of the first published version.

Existing APIs are unchanged. In particular, `session.get_ground_truth(
dataset_name=...)` still returns every ground truth row of the dataset, so
publishing a version of a subset does not silently change what existing callers
read. Pass `dataset_version_id` to read a pinned snapshot in the same
`GroundTruth` shape:

```python
df = session.get_ground_truth(
    dataset_version_id=version.dataset_version_id,
)
```

# Golden Set Generation

The hardest part of aligning LLM judges is building a good golden set. TruLens
already stores production records — inputs, outputs and judge scores — in its
database, so instead of hand-writing lists of dicts you can sample real traffic
for human annotation.

`GoldenSetGenerator` queries a `TruSession` for records, samples them with a
configurable strategy, and exports the sample in the `GroundTruthAgreement`
golden-set format or to CSV/JSON for external annotation tools. Once
annotated, the golden set is validated, loaded back, and persisted via
`TruSession.add_ground_truth_to_dataset`.

!!! Initial Setup

    ```bash
    pip install trulens-benchmark
    ```

## Sampling

```python
from trulens.benchmark.golden_set_generator import GoldenSetGenerator
from trulens.core import TruSession

session = TruSession()
generator = GoldenSetGenerator(session, seed=42)

# Sample 50 records, stratified by existing relevance scores.
sample = generator.sample(
    n=50,
    app_name="my_rag_app",
    strategy="stratified",
    feedback_name="relevance",
)

# Export for annotation.
sample.to_csv("golden_set_to_annotate.csv")
```

Three sampling strategies are supported:

- **`random`** — uniform random sample. No judge scores needed.
- **`stratified`** — equal samples from the low (`< 1/3`), medium and high
  (`>= 2/3`) buckets of an existing judge's scores, so the golden set covers
  clear failures, borderline cases and clear successes. Sparse buckets are
  backfilled from the rest of the pool.
- **`uncertainty`** — the records whose judge scores sit closest to the
  decision boundary (`0.5` by default, configurable via `decision_boundary`).
  These are the examples the judge is least sure about and where human labels
  add the most signal.

`stratified` and `uncertainty` require `feedback_name` — the name of the
feedback function whose scores drive the sampling.

Records can also be filtered before sampling: by app (`app_name`,
`app_version`), by date range (`start_time`, `end_time`) or by judge score
range (`min_score`, `max_score`). Pass `seed` to the constructor for a
reproducible sample.

## Annotation

The exported CSV/JSON contains a `query`, `expected_response` and an empty
`expected_score` column per row, along with provenance (record id, app
name/version, timestamp, and the sampled judge's score) so annotators have
context. Fill in `expected_score` — and correct `expected_response` where the
production output was wrong — with any external tool, then load the file back:

```python
annotated = generator.load_annotations("golden_set_annotated.csv")
```

`load_annotations` validates that every row has a non-empty `query` and a
numeric `expected_score` within `[0, 1]` (both configurable via
`require_scores` and `score_range`).

For quick in-notebook use, `sample.to_list()` returns the sample directly in
the `GroundTruthAgreement` golden-set format:

```python
[{"query": "...", "expected_response": "...", "expected_score": None}, ...]
```

## Persisting

```python
generator.save_golden_set("my_golden_set", annotated)
```

This stores the golden set as a TruLens dataset via
`session.add_ground_truth_to_dataset`, with `expected_score` (and the source
`record_id`) kept in each row's `meta` so they survive the round trip. Load it
later with `session.get_ground_truth("my_golden_set")` and feed it to
`GroundTruthAgreement` or the alignment utilities in `trulens-benchmark`.

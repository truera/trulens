# trulens-connectors-openeval

Convert between TruLens `Run` records and [EvalPort](https://github.com/adhabnr-ux/evalport) (Apache 2.0), the open interchange format for portable LLM evaluation datasets -- test cases, graders, suites, and results as plain JSON, shared across DeepEval, Promptfoo, Inspect AI, AutoGen, CrewAI, Ragas, LangSmith, Braintrust, MLflow, and Opik.

See [truera/trulens#2680](https://github.com/truera/trulens/issues/2680) for the proposal this module implements.

> **Status:** approved by @joshreini1 on 2026-08-17 ("Ready to merge" -- the two failing checks are pre-existing `test_otel_async_concurrency.py` flakes, unrelated to this change).

## Install

This module ships inside the `trulens` monorepo as `trulens-connectors-openeval`, following the same layout as `trulens-hotspots` and `src/connectors/snowflake`:

```bash
pip install trulens-connectors-openeval
```

## Usage

### Export a completed run's records + feedback scores to an EvalPort ResultSet

```python
from trulens.connectors.openeval import to_openeval
from openeval.validate import validate_result_set

records_df = run.get_records()  # record_id, input, output, latency, + one column per feedback
records_df.attrs["run_name"] = run.run_name  # optional, becomes ResultSet.run_id

result_set = to_openeval(records_df, suite_id="my_eval_suite")
assert validate_result_set(result_set).valid

import json
with open("results.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

Each TruLens feedback column (`Context Relevance`, `Groundedness`, `LogicalConsistency`, `ToolSelection`, etc.) becomes its own EvalPort `GraderResult` (`type: "custom"`, `grader_id` normalized from the feedback name, e.g. `"Context Relevance"` -> `"context_relevance"`). TruLens's `<name>_calls` companion columns (per-call detail, not a score) are automatically excluded -- pass `metric_columns` explicitly if you want to override the auto-detected set. A result's overall `passed` follows the same convention every other EvalPort adapter uses: every one of its grader results must individually pass `pass_threshold` (default `0.5`).

### Load an EvalPort suite as TruLens run input rows

```python
from trulens.connectors.openeval import from_openeval
from trulens.core.run import RunConfig

input_df, dataset_spec = from_openeval(suite)

run_config = RunConfig(
    run_name="from_evalport_suite",
    dataset_name="my_dataset",
    dataset_spec=dataset_spec,
)
run = my_app.add_run(run_config=run_config)
run.start(input_df=input_df)
```

`dataset_spec` maps TruLens's reserved dataset fields (`input`, `ground_truth_output`, `input_id`) to `input_df`'s own column names, validated against TruLens's own `validate_dataset_spec` / `get_all_span_attribute_key_constants` reserved vocabulary -- so it's ready to hand straight to `RunConfig` without further translation.

## Why this converts DataFrames, not `Run`/`RunConfig` objects directly

`trulens.core.run.Run` requires a live `RunDaoBase`, `TruSession`, and app instance just to construct -- it's inherently tied to a running TruLens session with a backing database, not a portable, serializable object. The DataFrames `Run.get_records()`/`Run.get_record_details()` return (and `Run.start()` accepts) are the actual portable surface, so this module converts at that boundary. This also means it can be imported and unit-tested without spinning up a live TruSession -- see `tests/unit/test_openeval.py`.

## What round-trips losslessly, and what doesn't

Every field this module explicitly maps (`input`, `ground_truth_output`, `input_id`, `record_id`, `output`, `latency`) round-trips cleanly. What doesn't survive a round-trip through a *different* EvalPort-speaking tool is TruLens-specific semantics -- span attributes, cost, per-call args -- which live in EvalPort's free-form `metadata["trulens"]` rather than being dropped, the same tradeoff every adapter in the EvalPort ecosystem makes (see e.g. the [Opik adapter](https://github.com/adhabnr-ux/evalport/tree/main/adapters/opik-openeval-adapter)'s README for the same pattern).

## Spec

<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>

## License

MIT -- see the repository LICENSE.

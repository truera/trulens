---
categories:
  - General
date: 2026-04-30
---

# TruLens 2.8: Parallel Batch Evals, Schema Validation, and a Faster Dashboard

TruLens 2.8 adds parallel batch evals on every backend (up to 5.4x speedup), programmatic schema validation for structured output, and SQL-level dashboard aggregation that keeps things responsive at 10k+ records. TruSession init is 73% faster and silent by default.

<!-- more -->

---

## Parallel Batch Evaluation, Now on Any Backend

The Run API was Snowflake-only. Now `run.start()`, `run.compute_metrics()`, and `run.get_records()` work with any connector (SQLite, PostgreSQL, Snowflake) and run in parallel.

### New RunConfig Parameters

Two knobs for concurrency control:

- `invocation_max_workers`: threads for `run.start()` (default: `min(len(input_df), 4)`)
- `metric_max_workers`: threads for `run.compute_metrics()` (default: `len(metrics)`)

```python
from trulens.core.run import RunConfig

config = RunConfig(
    run_name="batch_eval_v1",
    dataset_name="eval_questions",
    source_type="TABLE",
    dataset_spec={"input": "QUESTION"},
    invocation_max_workers=8,   # parallel app calls
    metric_max_workers=4,       # parallel metric computation
)

run = tru_app.add_run(run_config=config)
run.start()  # invokes app on all rows in parallel
run.compute_metrics(["groundedness", my_custom_metric])
```

### Benchmark: OSS (SQLite, Claude Sonnet, 8 questions, 4 metrics)

| Step | Sequential (workers=1) | Parallel (workers=4) | Speedup |
|------|------------------------|---------------------|---------|
| `run.start()` | 15.79s | 5.36s | **2.95x** |
| `run.compute_metrics()` | 384.89s | 145.95s | **2.64x** |

### Benchmark: Snowflake (client-side metrics, 4 LLM-as-judge evals)

| Step | Sequential | Parallel (workers=4) | Speedup |
|------|-----------|---------------------|---------|
| `run.compute_metrics()` | 417.85s | 77.83s | **5.37x** |

This resolves 6+ long-standing issues: nested recording errors ([#2325](https://github.com/truera/trulens/discussions/2325)), background feedback polling ([#2335](https://github.com/truera/trulens/discussions/2335)), and rate-limit control ([#687](https://github.com/truera/trulens/issues/687)).

**Docs:** [Batch Evaluation](https://www.trulens.org/component_guides/evaluation/batch_evaluation/)

---

## SchemaValidator: Programmatic Output Validation

The new `SchemaValidator` checks output against a Pydantic model or JSON schema dict. It's a pure programmatic check that plugs into the Metric API like any other feedback function.

### Pydantic model

```python
import pydantic
from trulens.feedback.schema_validator import SchemaValidator
from trulens.core.metric.metric import Metric

class ToolCall(pydantic.BaseModel):
    tool_name: str
    arguments: dict
    reasoning: str

validator = SchemaValidator(schema=ToolCall)
f_schema = Metric(validator.validate_json).on_output()
```

### JSON schema dict

```python
from trulens.feedback.schema_validator import SchemaValidator
from trulens.core.metric.metric import Metric

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer", "confidence"],
}

validator = SchemaValidator(schema=schema)
f_schema = Metric(validator.validate_json).on_output()
```

Returns `1.0` (valid) or `0.0` (invalid) with an `"explanation"` in metadata. Also ships `validate_json_partial` for streaming/partial output.

Pydantic validation needs no extra deps. JSON schema dict mode requires the optional `jsonschema` package.

---

## Dashboard Performance: Up to 5.2x Faster

The leaderboard used to fetch every record with full JSON payloads, deserialize in Python, then `groupby` in pandas. At 10k+ records that meant 20-30s loads.

Now aggregation is pushed into SQL via SQLAlchemy's cross-dialect query builder. The leaderboard fetches only pre-aggregated rows.

### Benchmark (10k records, SQLite, 5 runs each)

| Scenario | Old (Python agg) | New (SQL agg) | Speedup |
|----------|-----------------|---------------|---------|
| All apps, 15 versions | 1.330s | 0.255s | **5.2x** |
| Single app, 3 versions | 0.309s | 0.114s | **2.7x** |
| Single app, limit=1000 | 0.194s | 0.118s | **1.6x** |
| Single version | 0.163s | 0.090s | **1.8x** |

Also:

- New indexes on `start_timestamp` and `timestamp`
- Histogram tab lazy-loads raw records only when selected
- Sort order fixed (newest first)
- EVAL_ROOT metric name parsing fix for Compare page

---

## Fast and Quiet TruSession Startup

`TruSession()` took ~1.6s to init (eager provider imports in `_track_costs()`) and printed 6 lines to stdout. Both fixed.

### Background cost tracking

`_track_costs()` now runs in a daemon thread. On first span, `on_start` joins it, but it's usually done by then.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Init time | 1.63s | 0.44s | **73% faster** |
| Total (import + init) | 2.78s | 1.62s | **42% faster** |

First-span latency vs. time since init:

| Delay after init | Extra latency |
|------------------|---------------|
| 0ms | 1.14s |
| 500ms | 0.66s |
| 1.0s | 0.22s |
| 1.5s+ | 0s |

### Silent by default

All prints converted to `logger.info`/`logger.debug` or suppressed. Use `logging.basicConfig(level=logging.INFO)` to see the db URL.

---

## New Example: OpenAI Agent SDK + Snowflake Tools

A production reference for building an observable agentic app:

- OpenAI Agent SDK with Cortex Analyst + Cortex Search tools
- Batch eval runner using the Run API
- FastAPI monitoring backend
- React chat UI
- Streamlit observability dashboard

**Code:** [openai_agent_sdk_snowflake_tools](https://github.com/truera/trulens/tree/main/examples/expositional/frameworks/openai_agent_sdk/openai_agent_sdk_snowflake_tools/)

---

## Bug Fixes

- **Arrow-backed DataFrame fix:** `run.compute_metrics` converts Arrow columns to object dtype ([#2387](https://github.com/truera/trulens/pull/2387))
- **Duplicate Alembic revision:** Fixed conflicting revision 11, relaxed `importlib-resources` bound ([#2429](https://github.com/truera/trulens/pull/2429))
- **CI pipeline timeouts:** Added pip dependency caching ([#2395](https://github.com/truera/trulens/pull/2395))
- **Testset generation docs:** Fixed imports and install instructions for `trulens.benchmark.generate` ([#2384](https://github.com/truera/trulens/pull/2384))

---

## Get Started

```bash
pip install trulens --upgrade
```

### Links

- [Documentation](https://www.trulens.org/)
- [GitHub](https://github.com/truera/trulens)
- [Batch Evaluation Guide](https://www.trulens.org/component_guides/evaluation/batch_evaluation/)
- [OpenAI Agent SDK Example](https://github.com/truera/trulens/tree/main/examples/expositional/frameworks/openai_agent_sdk/openai_agent_sdk_snowflake_tools/)
- [Full Changelog](https://github.com/truera/trulens/compare/trulens-2.7.2...trulens-2.8.0)

---

Questions or feedback? Open an [issue](https://github.com/truera/trulens/issues) or [discussion](https://github.com/truera/trulens/discussions).

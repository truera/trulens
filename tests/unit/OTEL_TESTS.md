# OTEL Tests - Known Test Isolation Issue

## Problem

OTEL instrumentation tests (`test_otel*.py`) have a test isolation issue when run together in a single process:
- **6 tests fail** when run in batch: `pytest tests/unit/test_otel*.py`
- **All tests pass** when run individually or by file

## Root Cause

The OTEL `BatchSpanProcessor` writes spans asynchronously in background threads. Even after calling `force_flush()`, there's a race condition where:

1. Test A finishes and calls `force_flush()`
2. Test setup resets the database
3. Background threads from Test A are still writing spans
4. Test B starts and creates new spans
5. Result: Database contains spans from both tests (e.g., 14 expected, 24 actual)

Additional complexity:
- Evaluator threads continue running in background
- TruSession singleton caches database connections
- Class-level instrumentation flags persist across tests

## Solution: Process Isolation

The **only reliable solution** is to run each test class in a separate process using `pytest-xdist`.

### For CI/CD (Azure Pipelines / Jenkins)

Ensure `pytest-xdist` is installed and use this command pattern:

```bash
# Install dev dependencies (includes pytest-xdist)
poetry install --with dev

# Run OTEL tests with process isolation
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel*.py -n auto --dist=loadscope
```

Or use the Makefile which handles this automatically:

```bash
TEST_OPTIONAL=1 make test-unit
```

### For Local Development

**Option 1: Run individual test files (always works)**
```bash
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel_inline_evaluations.py -v
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel_tru_chain.py -v
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel_tru_graph.py -v
```

**Option 2: Use pytest-xdist (requires `poetry install --with dev`)**
```bash
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel*.py -n auto --dist=loadscope
```

**Option 3: Use Makefile**
```bash
make test-unit  # Automatically uses isolation when pytest-xdist is available
```

## Affected Tests

When run in batch without isolation, these 6 tests fail due to database pollution:

1. `test_otel_inline_evaluations.py::TestOtelInlineEvaluations::test_emitted_spans`
2. `test_otel_inline_evaluations.py::TestOtelInlineEvaluations::test_unemitted_spans`
3. `test_otel_tru_chain.py::TestOtelTruChain::test_legacy_app`
4. `test_otel_tru_chain.py::TestOtelTruChain::test_smoke`
5. `test_otel_tru_graph.py::TestOtelTruGraph::test_smoke`
6. `test_otel_tru_graph.py::TestOtelTruGraph::test_task_instrumentation`

**All pass when run individually** ✅

## Implementation Details

### What We Tried (And Why It Didn't Work)

1. ❌ **Resetting instrumentation flags** - Wrappers are already applied to classes
2. ❌ **Clearing TruSession singleton** - Background threads still writing
3. ❌ **Longer sleep after flush** - Race condition persists
4. ❌ **Unique database per test class** - Singleton initialized before env var set
5. ❌ **Stopping evaluator threads** - Too aggressive, broke other tests
6. ✅ **Process isolation via pytest-xdist** - Only reliable solution

### pytest-xdist Configuration

The Makefile uses:
```makefile
PYTEST_ISOLATED := poetry run pytest --rootdir=. -s -r fex --durations=0 -n auto --dist=loadscope
```

- `-n auto`: Uses one worker per CPU core
- `--dist=loadscope`: Ensures each test class runs in the same worker (maintains class-level setup/teardown)

This guarantees complete isolation between test classes while preserving test class semantics.

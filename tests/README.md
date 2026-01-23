# TruLens Test Suite

## Running Tests

### All Tests
```bash
make test-unit
```

### OTEL Tests (Require Isolation)

OTEL tests use async background threads (OTEL BatchSpanProcessor) that can cause test pollution when tests run in the same process. The Makefile is configured to run these tests with process isolation using `pytest-xdist`.

**Prerequisites:**
```bash
poetry install --with dev  # Installs pytest-xdist
```

**Run OTEL tests individually (always works):**
```bash
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel_inline_evaluations.py -v
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel_tru_chain.py -v
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel_tru_graph.py -v
# ... etc
```

**Run OTEL tests with isolation (requires pytest-xdist):**
```bash
TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel*.py -n auto --dist=loadscope
```

**Why isolation is needed:**
1. OTEL spans are written asynchronously by background threads
2. `force_flush()` signals the batch processor but doesn't guarantee completion
3. Tests may see events from previous tests even after database reset
4. Running each test file or test class in a separate process guarantees isolation

### Test Markers

- `@pytest.mark.optional`: Requires optional dependencies (LangChain, LlamaIndex, LangGraph)
- `@pytest.mark.snowflake`: Requires Snowflake credentials
- `@pytest.mark.huggingface`: Requires HuggingFace API access

Enable optional tests:
```bash
TEST_OPTIONAL=1 make test-unit
```

## Golden Files

Some tests use golden files (CSV snapshots) for regression testing. To regenerate:

```bash
WRITE_GOLDEN=1 TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel_tru_chain.py::TestOtelTruChain::test_smoke
```

**Important:** Only regenerate golden files when you've verified the new behavior is correct!

## Troubleshooting

### "6 tests fail when run together but pass individually"
This is expected if `pytest-xdist` is not installed. Install it with:
```bash
poetry install --with dev
```

Then run via Makefile which uses proper isolation:
```bash
make test-unit
```

### "Database has events from previous test"
Use `make test-unit` which handles isolation, or run test files individually.

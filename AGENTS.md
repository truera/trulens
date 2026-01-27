# AGENTS.md

## Setup commands

- Install deps: `poetry install`
- Install with dev tools: `poetry install --with dev`
- Install with all optional packages: `poetry install --with dev,apps,providers`
- Install pre-commit hooks: `pre-commit install`
- Start documentation server: `make docs-serve`

## Code style

- Line length: 80 characters
- Formatter/linter: `ruff` (run `make format` and `make lint`)
- Google-style docstrings
- Import modules, not classes directly:
  ```python
  # ✓ Do this
  from trulens.schema import record as record_schema
  from trulens.providers.openai import provider as openai_provider

  # ✗ Not this
  from trulens.schema.record import Record
  ```
- Standard module rename patterns:
  ```python
  from trulens.schema import X as X_schema
  from trulens.utils import X as X_utils
  from trulens.providers.X import provider as X_provider
  from trulens.apps.X import Y as Y_app
  from trulens.core import X as core_X
  from trulens.core.database import base as core_db
  ```
- Use `TYPE_CHECKING` blocks for type-only imports
- Use `from __future__ import annotations` for forward references
- Call `model_rebuild()` after Pydantic models with forward refs

## Testing instructions

- Run all unit tests: `make test-unit`
- Run single test file: `TEST_OPTIONAL=true poetry run pytest tests/unit/test_file.py -v`
- Run specific test: `TEST_OPTIONAL=true poetry run pytest tests/unit/test_file.py::TestClass::test_method`
- OTEL tests require isolation (uses pytest-xdist): `TEST_OPTIONAL=1 poetry run pytest tests/unit/test_otel*.py -n auto --dist=loadscope`
- Regenerate golden files: `WRITE_GOLDEN=1 TEST_OPTIONAL=1 poetry run pytest <test_path>`

### Test markers

- `@pytest.mark.optional` - requires optional dependencies
- `@pytest.mark.snowflake` - requires Snowflake credentials
- `@pytest.mark.huggingface` - requires HuggingFace access

Enable optional tests: `TEST_OPTIONAL=true`

## Build commands

- Format code: `make format`
- Lint code: `make lint`
- Build all packages: `make build`
- Build docs: `make docs`
- Update poetry locks: `make lock`
- Generate coverage: `make coverage`

## Project structure

```
src/
├── core/           # trulens-core: Core abstractions, session, database
├── feedback/       # trulens-feedback: Feedback function implementations
├── dashboard/      # trulens-dashboard: Streamlit UI + React components
├── apps/           # App integrations (langchain, langgraph, llamaindex)
├── providers/      # LLM providers (openai, bedrock, cortex, huggingface, litellm)
├── connectors/     # Database connectors (snowflake)
└── otel/semconv/   # OpenTelemetry semantic conventions
```

## Key patterns

### TruSession (main entry point)
```python
from trulens.core import TruSession
session = TruSession()  # Default SQLite
session = TruSession(database_url="postgresql://...")
```

### App wrappers
```python
from trulens.apps.langchain import TruChain
tru_app = TruChain(chain, app_name="MyApp", app_version="v1", feedbacks=[...])
with tru_app as recording:
    result = chain.invoke("query")
```

### OTEL instrumentation

Basic instrumentation - captures function args and return as span attributes:
```python
from trulens.core.otel.instrument import instrument

@instrument()
def my_function():
    pass  # Automatically traced
```

#### Span types

Use `span_type` to categorize spans for semantic meaning:
```python
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

@instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
def retrieve(self, query: str) -> list:
    pass

@instrument(span_type=SpanAttributes.SpanType.GENERATION)
def generate(self, prompt: str) -> str:
    pass
```

Available span types: `RETRIEVAL`, `GENERATION`, `RERANKING`, `TOOL`, `AGENT`, `WORKFLOW`, `GRAPH_NODE`, `GRAPH_TASK`, `MCP`, `RECORD_ROOT`, `EVAL_ROOT`, `UNKNOWN`

#### Custom span attributes

Map function args/return to semantic attributes:
```python
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",      # maps "query" arg
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",  # maps return value
    },
)
def retrieve(self, query: str) -> list:
    return ["context1", "context2"]
```

Common attribute namespaces:
- `SpanAttributes.RETRIEVAL` - `QUERY_TEXT`, `RETRIEVED_CONTEXTS`, `NUM_CONTEXTS`
- `SpanAttributes.RECORD_ROOT` - `INPUT`, `OUTPUT`, `ERROR`
- `SpanAttributes.MCP` - `TOOL_NAME`, `SERVER_NAME`, `INPUT_ARGUMENTS`, `OUTPUT_CONTENT`
- `SpanAttributes.RERANKING` - `QUERY_TEXT`, `MODEL_NAME`, `TOP_N`, `INPUT_CONTEXT_TEXTS`

#### Manipulating attributes with lambdas

For complex data extraction, use a lambda with signature `(ret, exception, *args, **kwargs)`:
```python
@instrument(
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [doc["text"] for doc in ret],
        SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs["query"].upper(),
    }
)
def retrieve_contexts(self, query: str) -> list:
    return [{"text": "ctx1", "source": "doc.pdf"}, {"text": "ctx2", "source": "doc2.pdf"}]
```

Lambda parameters:
- `ret` - function return value
- `exception` - any exception raised (None if successful)
- `*args` - positional arguments
- `**kwargs` - keyword arguments (includes positional args by name)

#### Instrumenting third-party classes

Use `instrument_method()` when you can't modify source code:
```python
from trulens.core.otel.instrument import instrument_method
from somepackage import CustomRetriever

instrument_method(
    cls=CustomRetriever,
    method_name="retrieve",
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    }
)
```

### Evaluation

Feedback functions evaluate LLM app quality. Must return `float` in [0.0, 1.0] or `dict[str, float]`.

#### Basic usage with shortcuts
```python
from trulens.core import Feedback
from trulens.providers.openai import OpenAI

provider = OpenAI()
f_relevance = Feedback(provider.relevance_with_cot_reasons).on_input().on_output()
```

Shortcuts:
- `on_input()` - selects `RECORD_ROOT.INPUT`
- `on_output()` - selects `RECORD_ROOT.OUTPUT`
- `on_context()` - selects `RETRIEVAL.RETRIEVED_CONTEXTS`

#### Selecting span attributes with Selector

Use `Selector` to explicitly select instrumented span attributes for evaluation:
```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import SpanAttributes

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on({
        "prompt": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
    })
    .on({
        "response": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        ),
    })
)
```

#### Using collect_list for retrieved contexts

- `collect_list=False` - evaluate each context individually (for context relevance)
- `collect_list=True` - concatenate all contexts for single evaluation (for groundedness)

```python
# Evaluate each retrieved context individually
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=False
        ),
    })
)

# Evaluate groundedness against all contexts combined
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True
        ),
    })
    .on_output()
)
```

### Experimental features
```python
from trulens.core.experimental import Feature
session = TruSession(experimental_feature_flags=[Feature.OTEL_TRACING])
```

## Adding new components

### New provider
1. Create `src/providers/<name>/` with `pyproject.toml`
2. Extend `trulens.feedback.LLMProvider`
3. Implement `_create_chat_completion(self, prompt, messages, **kwargs)`
4. Add endpoint class for API interactions

### New app integration
1. Create `src/apps/<name>/` with `pyproject.toml`
2. Extend `trulens.core.app.App`
3. Define `Default.CLASSES` and `Default.METHODS` for instrumentation
4. Implement `main_input()` and `main_output()` methods

### New feedback function
1. Add method to provider class or `LLMProvider`
2. Return float [0, 1] or dict of floats
3. Add Google-style docstring with example
4. Add tests

## Troubleshooting

- **Circular imports**: Use `from __future__ import annotations` and `TYPE_CHECKING` blocks
- **OTEL tests failing in batch**: Install pytest-xdist (`poetry install --with dev`) and use `make test-unit`
- **Missing optional deps**: TruLens uses lazy imports - install specific packages as needed

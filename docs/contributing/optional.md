# ⛅ Optional Packages

Since TruLens 1.0, the package has been modularized into a family of separate
installable packages. This design minimizes the overhead for developers to use
only the capabilities they need.

## Package Architecture

The TruLens ecosystem is split into the following packages:

- **`trulens-core`**: Core abstractions for database operations, app
  instrumentation, guardrails and evaluation.
- **`trulens-dashboard`**: Capabilities to run and operate the TruLens
  dashboard.
- **`trulens-feedback`**: Out-of-the-box feedback function implementations.
  Must be combined with a provider integration.

### App Integrations (`trulens-apps-*`)

Tools for instrumenting LLM apps built with other frameworks:

- `trulens-apps-langchain` - Instrument LangChain apps with `TruChain`
- `trulens-apps-llamaindex` - Instrument LlamaIndex apps with `TruLlama`
- `trulens-apps-nemo` - Instrument NeMo Guardrails apps

### Provider Integrations (`trulens-providers-*`)

Integrations for running feedback functions with different LLM providers:

- `trulens-providers-openai` - OpenAI models
- `trulens-providers-litellm` - LiteLLM (supports Ollama and many others)
- `trulens-providers-huggingface` - HuggingFace models
- `trulens-providers-bedrock` - AWS Bedrock models
- `trulens-providers-cortex` - Snowflake Cortex models
- `trulens-providers-langchain` - LangChain-wrapped models

### Database Connectors (`trulens-connectors-*`)

Ways to log TruLens traces and evaluations to different databases:

- `trulens-connectors-snowflake` - Connect to Snowflake

## Installation Examples

Install only what you need:

```bash
# Basic installation (includes core, feedback, and dashboard)
pip install trulens

# For LangChain apps with OpenAI feedback
pip install trulens-apps-langchain trulens-providers-openai

# For LlamaIndex apps with local models via LiteLLM
pip install trulens-apps-llamaindex trulens-providers-litellm

# For logging to Snowflake
pip install trulens-connectors-snowflake
```

## Dev Notes

### Internal Optional Imports

Within individual TruLens packages, we still use an `OptionalImports`
context-manager-based scheme (see `trulens.core.utils.imports`) to handle
imports that may not be installed. This is primarily used for:

1. Optional dependencies within a package (e.g., dashboard widgets)
2. Graceful error messages when a required sibling package is missing

Example usage within TruLens code:

```python
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.imports import format_import_errors

with OptionalImports(messages=format_import_errors("some-optional-package")):
    import some_optional_package
```

This makes it so that `some_optional_package` gets defined even if the import
fails. However, if the user tries to use it, they will be presented with a
helpful message explaining how to install the missing package.

### When to Fail

Imports from a general package that does not imply an optional package should
not produce an error immediately. However, imports from packages that do imply
the use of an optional dependency should fail immediately with a helpful
message.

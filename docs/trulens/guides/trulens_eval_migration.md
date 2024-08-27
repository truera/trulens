
# Moving from `trulens-eval`

This document highlights the changes required to move from `trulens-eval` to `trulens`.

The biggest change is that the `trulens` library now consists of several interoperable modules, each of which can be installed and used independently. This allows users to mix and match components to suit their needs without needing to install the entire library.

When running `pip install trulens`, the following base modules are installed:

- `trulens-core`: core module that provides the main functionality for TruLens.
- `trulens-feedback`: The module that provides LLM-based evaluation and feedback function definitions.
- `trulens-dashboard`: The module that supports the streamlit dashboard and evaluation visualizations.

Furthermore, the following additional modules can be installed separately:
- `trulens-benchmark`: provides benchmarking functionality for evaluating feedback functions on your dataset.

Instrumentation libraries used to instrument specific frameworks like LangChain and LlamaIndex are now packaged separately and imported under the `trulens.instrument` namespace. For example, to use TruChain to instrument a LangChain app, run `pip install trulens-instrument-langchain` and import it as follows:

```python
from trulens.instrument.langchain import TruChain
```
Similarly, providers are now packaged separately from the core library. To use a specific provider, install the corresponding package and import it as follows:

```python
from trulens.providers.openai import OpenAI
```

To find a full list of providers, please refer to the [API Reference][trulens.providers.cortex].


## Common Import Changes

As a result of these changes, the package structure for the TruLens varies from TruLens-Eval. Here are some common import changes you may need to make:

| TruLens Eval | TruLens | Additional Dependencies |
|------------|-------------|------------------|
| `trulens_eval.Tru` | [`trulens.core.TruSession`][trulens.core.TruSession] | |
| `trulens_eval.Feedback` | [`trulens.core.Feedback`][trulens.core.Feedback] | |
| `trulens_eval.Select` | [`trulens.core.Select`][trulens.core.Select] | |
| `trulens_eval.TruCustomApp`, `TruSession().Custom(...)` | [`trulens.instrument.custom.TruCustomApp`][trulens.instrument.custom.TruCustomApp] | |
| `trulens_eval.TruChain`, `TruSession().Chain(...)` | [`trulens.instrument.langchain.TruChain`][trulens.instrument.langchain.TruChain] | `trulens-instrument-langchain` |
| `trulens_eval.TruLlama`, `TruSession().Llama(...)` | [`trulens.instrument.llamaindex.TruLlama`][trulens.instrument.llamaindex.TruLlama] | `trulens-instrument-llamaindex` |
| `trulens_eval.OpenAI` | [`trulens.providers.openai.OpenAI`][trulens.providers.openai.OpenAI] | `trulens-providers-openai` |
| `trulens_eval.Huggingface` | [`trulens.providers.huggingface.Huggingface`][trulens.providers.huggingface.Huggingface] | `trulens-providers-huggingface` |
| `trulens_eval.guardrails.llama` | [`trulens.instrument.llamaindex.guardrails`][trulens.instrument.llamaindex.guardrails] | `trulens-instrument-llamaindex` |
| `TruSession().run_dashboard()` | [`trulens.dashboard.run_dashboard()`][trulens.dashboard.run_dashboard] | `trulens-dashboard` |

To find a specific definition, use the search functionality or go directly to the [API Reference](../../reference/trulens/core/index.md).

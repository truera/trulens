
# Moving from `trulens-eval`

This document highlights the changes required to move from `trulens-eval` to `trulens`.

The biggest change is that the `trulens` library now consists of several interoperable modules, each of which can be installed and used independently. This allows users to mix and match components to suit their needs without needing to install the entire library.

When running `pip install trulens`, the following base modules are installed:

- `trulens-core`: core module that provides the main functionality for TruLens.
- `trulens-feedback`: The module that provides LLM-based evaluation and feedback function definitions.
- `trulens-dashboard`: The module that supports the streamlit dashboard and evaluation visualizations.

Furthermore, the following additional modules can be installed separately:
- `trulens-benchmark`: provides benchmarking functionality for evaluating feedback functions on your dataset.

Instrumentation libraries used to instrument specific frameworks like LangChain and LlamaIndex are now packaged separately and imported under the `trulens.apps` namespace. For example, to use TruChain to instrument a LangChain app, run `pip install trulens-apps-langchain` and import it as follows:

```python
from trulens.apps.langchain import TruChain
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
| `trulens_eval.Tru` | [trulens.core.TruSession][trulens.core.TruSession] | |
| `trulens_eval.Feedback` | [trulens.core.Feedback][trulens.core.Feedback] | |
| `trulens_eval.Select` | [trulens.core.Select][trulens.core.Select] | |
| `trulens_eval.TruCustomApp`, `TruSession().Custom(...)` | [trulens.apps.custom.TruCustomApp][trulens.apps.custom.TruCustomApp] | |
| `trulens_eval.TruChain`, `Tru().Chain(...)` | [`TruSession().App(...)`][trulens.core.session.TruSession.App] or [trulens.apps.langchain.TruChain][trulens.apps.langchain.TruChain] | `trulens-apps-langchain` |
| `trulens_eval.TruLlama`, `Tru().Llama(...)` | [`TruSession().App(...)`][trulens.core.session.TruSession.App] or [trulens.apps.llamaindex.TruLlama][trulens.apps.llamaindex.TruLlama] | `trulens-apps-llamaindex` |
| `trulens_eval.TruRails`, `Tru().Rails(...)` | [`TruSession().App(...)`][trulens.core.session.TruSession.App] or [trulens.apps.nemo.TruRails][trulens.apps.nemo.TruRails] | `trulens-apps-nemo` |
| `trulens_eval.OpenAI` | [trulens.providers.openai.OpenAI][trulens.providers.openai.OpenAI] | `trulens-providers-openai` |
| `trulens_eval.Huggingface` | [trulens.providers.huggingface.Huggingface][trulens.providers.huggingface.Huggingface] | `trulens-providers-huggingface` |
| `trulens_eval.guardrails.llama` | [trulens.apps.llamaindex.guardrails][trulens.apps.llamaindex.guardrails] | `trulens-apps-llamaindex` |
| `Tru().run_dashboard()` | [`trulens.dashboard.run_dashboard()`][trulens.dashboard.run_dashboard] | `trulens-dashboard` |

To find a specific definition, use the search functionality or go directly to the [API Reference](../../reference/trulens/core/index.md).

## Automatic Migration with Grit

To assist you in migrating your codebase to _TruLens_ to v1.0, we've published a `grit` pattern. You can migrate your codebase [online](https://docs.grit.io/patterns/library/trulens_eval_migration#migrate-and-use-tru-session), or by using `grit` on the command line.

To use on the command line, follow these instructions:

### Install `grit`

You can install the Grit CLI from NPM:
```bash
npm install --location=global @getgrit/cli
```
Alternatively, you can also install Grit with an installation script:
```bash
curl -fsSL https://docs.grit.io/install | bash
```

### Apply automatic changes

```bash
grit apply trulens_eval_migration
```

Be sure to audit its changes: we suggest ensuring you have a clean working tree beforehand.

---
categories:
  - General
date: 2026-04-14
---

# TruLens 2.7: Unified Metric API, MLflow Integration, and More

TruLens 2.7 brings a cleaner evaluation API, first-class MLflow integration, improved Snowflake support, and a growing library of examples—making it easier than ever to evaluate, iterate, and trust your AI applications.

<!-- more -->

---

## Unified Metric API

The headline feature of TruLens 2.7 is the **Unified Metric API**: a single `Metric` class that replaces both `Feedback` and `MetricConfig`. If you've been using either API, they continue to work with deprecation warnings—but the new `Metric` class is the recommended path forward.

### Why We Unified Them

`Feedback` was the original TruLens evaluation primitive, while `MetricConfig` emerged as a cleaner configuration-first alternative. Running both in parallel created confusion: which should I use? Do they behave the same? The new `Metric` class answers both questions with a single, consistent interface.

### What Changed

The new `Metric` class uses an explicit `selectors` dictionary instead of chained `.on()` calls, making the mapping from LLM arguments to span data clear at a glance:

!!! example "Before (Feedback API)"

    ```python
    from trulens.core import Feedback
    from trulens.providers.openai import OpenAI

    provider = OpenAI()

    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input()
        .on_output()
    )

    f_context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(Select.RecordCalls.retrieve.rets)
        .aggregate(numpy.mean)
    )
    ```

!!! example "After (Metric API)"

    ```python
    import numpy as np
    from trulens.core import Metric, Selector
    from trulens.providers.openai import OpenAI

    provider = OpenAI()

    f_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="Answer Relevance",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    f_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="Context Relevance",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )
    ```

Both the old `Feedback` and `MetricConfig` classes continue to work but emit deprecation warnings guiding you to migrate. The behavior is identical—this is a pure API unification with no functional changes.

**Learn more:** [Metric API Documentation](https://www.trulens.org/component_guides/evaluation/)

---

## MLflow Integration

TruLens 2.7 adds first-class support for using TruLens feedback functions as **MLflow scorers** via `mlflow.genai.evaluate` (requires MLflow 3.10+). This means you can run the RAG Triad, agent evaluations, and any custom TruLens metric directly inside your MLflow evaluation pipelines—no adapter code required.

### What You Can Evaluate

- **RAG scorers**: Groundedness, Context Relevance, Answer Relevance
- **Output scorers**: Coherence, Helpfulness, Sentiment
- **Agent trace scorers**: ToolSelection, ToolCalling, ToolQuality

!!! example "TruLens Scorers in MLflow"

    ```python
    import mlflow
    from trulens.providers.openai import OpenAI
    from trulens.feedback.v2.feedback import Groundedness, ContextRelevance

    provider = OpenAI()

    # Define scorers using TruLens feedback functions
    groundedness_scorer = Groundedness(provider=provider)
    context_relevance_scorer = ContextRelevance(provider=provider)

    with mlflow.start_run():
        results = mlflow.genai.evaluate(
            model=my_rag_app,
            data=eval_dataset,
            scorers=[groundedness_scorer, context_relevance_scorer],
        )
        print(results.tables["eval_results_table"])
    ```

This integration lets teams who already use MLflow for experiment tracking add TruLens's LLM-as-a-judge evaluations to their existing workflows without switching tools.

**See the example:** [MLflow + TruLens Scorers Notebook](https://github.com/truera/trulens/tree/main/examples/expositional/frameworks/mlflow/)

---

## LiteLLM Custom Endpoints

TruLens's LiteLLM provider now correctly forwards `api_base`, `api_key`, and other routing parameters to completion calls. Previously, these params were silently dropped, making it impossible to use self-hosted models (Ollama, vLLM, etc.) or custom OpenAI-compatible endpoints as feedback providers.

!!! example "Using Ollama as a Feedback Provider"

    ```python
    from trulens.providers.litellm import LiteLLM

    # Via direct kwarg
    provider = LiteLLM(
        model_engine="ollama/llama3.1",
        api_base="http://localhost:11434",
    )

    # Via environment variable (LiteLLM reads OLLAMA_API_BASE automatically)
    import os
    os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
    provider = LiteLLM(model_engine="ollama/llama3.1")

    # Via completion_kwargs
    provider = LiteLLM(
        model_engine="ollama/llama3.1",
        completion_kwargs={"api_base": "http://localhost:11434"},
    )
    ```

This fix (reported in [#1804](https://github.com/truera/trulens/issues/1804)) unlocks fully local evaluation pipelines—instrument your app with TruLens, run feedback functions against a local Ollama instance, and keep everything on-prem.

**Learn more:** [LiteLLM Provider Documentation](https://www.trulens.org/component_guides/evaluation/feedback_providers/#litellm)

---

## Feedback Templates Reorganization

The `trulens.feedback` package now has a cleaner, domain-based layout for feedback template classes. What was a 1,500-line monolith is now organized into focused modules:

| Module | Contents |
|---|---|
| `trulens.feedback.templates.rag` | `Groundedness`, `ContextRelevance`, `PromptResponseRelevance`, `Answerability`, `Comprehensiveness` |
| `trulens.feedback.templates.safety` | `Harmfulness`, `Toxicity`, `Maliciousness`, `Hate`, `Misogyny`, `Stereotypes` |
| `trulens.feedback.templates.quality` | `Coherence`, `Correctness`, `Conciseness`, `Sentiment`, `Helpfulness` |
| `trulens.feedback.templates.agent` | `ToolSelection`, `ToolCalling`, `ToolQuality`, `PlanAdherence`, `PlanQuality`, `LogicalConsistency` |

All existing imports continue to work—`prompts.py` and `v2/feedback.py` are backward-compatible shims. This reorganization makes it easier to find, extend, and contribute feedback templates.

---

## Snowflake Improvements

### Password-Free Authentication

`SnowflakeConnector` now supports password-free authentication methods directly—no need to pre-build a Snowpark session. The `externalbrowser` SSO flow is the new recommended approach:

!!! example "SSO Authentication"

    ```python
    from trulens.connectors.snowflake import SnowflakeConnector
    from trulens.core import TruSession

    connector = SnowflakeConnector(
        account="myorg-myaccount",
        user="my.name@company.com",
        authenticator="externalbrowser",
        database="TRULENS_DB",
        schema="PUBLIC",
        warehouse="COMPUTE_WH",
    )
    session = TruSession(connector=connector)
    ```

Key-pair and OAuth token authentication are also supported via the `private_key_file` and `token` parameters respectively.

### Snowsight Evaluations

Snowflake users should use the **AI Observability Evaluations** page in Snowsight rather than launching a local Streamlit dashboard. The `run_dashboard_sis` entrypoints are now deprecated with migration guidance. The Snowsight UI provides a fully managed, scalable view of your TruLens traces and evaluations without any local infrastructure.

### Accurate Cortex Cost Tracking

Cortex model cost tracking now uses **input/output split pricing** for all supported models, giving you accurate cost breakdowns that match Snowflake's billing. Previously, a single blended rate was used.

---

## New Example: Hybrid Search RAG with Qdrant

A new example notebook shows how to build and evaluate a **Hybrid Search RAG pipeline** using LangChain, Qdrant, and OpenAI—then evaluate it end-to-end with TruLens.

The pipeline combines dense embeddings and sparse (BM25) retrieval for higher-quality context selection, and the notebook walks through applying the full RAG Triad (Groundedness, Context Relevance, Answer Relevance) to measure quality.

**See the example:** [Hybrid Search RAG with LangChain and Qdrant](https://github.com/truera/trulens/tree/main/examples/expositional/vector_stores/qdrant/)

---

## Get Started

Ready to try TruLens 2.7?

!!! example "Install TruLens"

    ```bash
    pip install trulens --upgrade
    ```

### Quick Links

- [TruLens Documentation](https://www.trulens.org/)
- [GitHub Repository](https://github.com/truera/trulens)
- [Metric API Guide](https://www.trulens.org/component_guides/evaluation/)
- [MLflow Integration Example](https://github.com/truera/trulens/tree/main/examples/expositional/frameworks/mlflow/)
- [Snowflake Auth Documentation](https://www.trulens.org/component_guides/logging/where_to_log/log_in_snowflake/)

---
**Have feedback or feature requests?** Open an [issue](https://github.com/truera/trulens/issues) or [discussion](https://github.com/truera/trulens/discussions) on GitHub.
---

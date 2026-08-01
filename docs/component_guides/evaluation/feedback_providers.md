# Feedback Providers

TruLens constructs feedback functions by combining more general models, known as
the [**_feedback provider_**][trulens.core.feedback.Provider], and
[**_feedback implementation_**](./feedback_implementations/index.md) made up of
carefully constructed prompts and custom logic tailored to perform a particular
evaluation task.

This page documents the feedback providers available in _TruLens_.

There are three categories of such providers as well as provider combinations
that make use of one or more of these providers to offer additional feedback
function capabilities based on the constituent providers.

## Classification-based Providers

Some feedback functions rely on classification typically tailor-made for evaluation tasks, unlike LLM models.

- [Hugging Face provider][trulens.providers.huggingface.Huggingface]
  containing a variety of classification-based feedback functions runnable on the remote HuggingFace API.
  - [Hugging Face Local provider][trulens.providers.huggingface.HuggingfaceLocal]
  containing a variety of classification-based feedback functions runnable locally.
- [OpenAI provider][trulens.providers.openai.OpenAI] (and
  subclasses) features moderation feedback functions.

## Generation-based Providers

Providers which use large language models for feedback evaluation:

- [OpenAI provider][trulens.providers.openai.OpenAI] or
  [AzureOpenAI provider][trulens.providers.openai.AzureOpenAI]
- [Google provider][trulens.providers.google.Google]
- [Bedrock provider][trulens.providers.bedrock.Bedrock]
- [LiteLLM provider][trulens.providers.litellm.LiteLLM]
- [_LangChain_ provider][trulens.providers.langchain.Langchain]

Feedback functions common to these providers are found in the abstract class
[LLMProvider][trulens.feedback.LLMProvider].

### Using LiteLLM with a Custom Endpoint

The [LiteLLM provider][trulens.providers.litellm.LiteLLM] supports
[100+ models](https://docs.litellm.ai/docs/providers) through
[LiteLLM](https://github.com/BerriAI/litellm), including local models
served by Ollama.

When connecting to a model served at a custom URL (e.g. a remote Ollama
instance), there are three options:

!!! example "Specifying a custom base URL"

    === "Direct keyword argument"

        Pass `api_base` directly to the provider constructor:

        ```python
        from trulens.providers.litellm import LiteLLM

        provider = LiteLLM(
            model_engine="ollama/llama3.1:8b",
            api_base="http://my-ollama-host:11434",
        )
        ```

    === "Environment variable"

        Set the provider-specific environment variable and litellm
        will read it automatically. For Ollama, this is
        `OLLAMA_API_BASE`:

        ```python
        import os
        os.environ["OLLAMA_API_BASE"] = "http://my-ollama-host:11434"

        from trulens.providers.litellm import LiteLLM

        provider = LiteLLM(model_engine="ollama/llama3.1:8b")
        ```

        See the
        [litellm docs](https://docs.litellm.ai/docs/providers)
        for the environment variable names for each provider.

    === "Via completion_kwargs"

        Use `completion_kwargs` to pass any extra arguments to
        `litellm.completion()`:

        ```python
        from trulens.providers.litellm import LiteLLM

        provider = LiteLLM(
            model_engine="ollama/llama3.1:8b",
            completion_kwargs={
                "api_base": "http://my-ollama-host:11434",
            },
        )
        ```

### Using the OpenAI provider with a custom `base_url`

The [OpenAI provider][trulens.providers.openai.OpenAI] forwards constructor
kwargs to the official OpenAI Python client. That means you can point feedback
functions at any OpenAI-compatible Chat Completions endpoint by setting
`base_url` (and usually `api_key`).

Common options include self-hosted gateways (vLLM, Ollama with an OpenAI
shim), cloud gateways (OpenRouter, Together, Fireworks, DaoXE, and others),
or any reverse-proxy that speaks `/v1/chat/completions`.

!!! example "OpenAI-compatible custom endpoint"

    Pass the gateway's Chat Completions base URL and a model ID that endpoint
    accepts. Model IDs differ by provider — use a concrete ID from that
    provider's catalog (examples below are illustrative).

    ```python
    import os

    from trulens.providers.openai import OpenAI

    # Examples of base_url values (pick one):
    #   "https://openrouter.ai/api/v1"
    #   "https://api.together.xyz/v1"
    #   "https://api.fireworks.ai/inference/v1"
    #   "https://daoxe.com/v1"
    #   "http://localhost:8000/v1"  # vLLM / local OpenAI shim
    provider = OpenAI(
        model_engine="gpt-4o-mini",  # or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", etc.
        api_key=os.environ["OPENAI_API_KEY"],  # gateway token / key
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # Example feedback call
    score, reasons = provider.relevance_with_cot_reasons(
        "What is the capital of France?",
        "Paris is the capital of France.",
    )
    ```

    Equivalent LiteLLM route (useful when you already use
    `trulens-providers-litellm`):

    ```python
    import os

    from trulens.providers.litellm import LiteLLM

    provider = LiteLLM(
        model_engine="openai/gpt-4o-mini",
        api_base=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    ```

## Embedding-based Providers

- [Embeddings][trulens.feedback.embeddings.Embeddings]

## Provider Combinations

- [GroundTruth][trulens.feedback.groundtruth.GroundTruthAgreement]

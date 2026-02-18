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

## Embedding-based Providers

- [Embeddings][trulens.feedback.embeddings.Embeddings]

## Provider Combinations

- [GroundTruth][trulens.feedback.groundtruth.GroundTruthAgreement]

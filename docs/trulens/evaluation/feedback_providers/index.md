# Feedback Providers

TruLens constructs feedback functions by combining more general models, known as
the [**_feedback provider_**][trulens.core.feedback.Provider], and
[**_feedback implementation_**](../feedback_implementations/index.md) made up of
carefully constructed prompts and custom logic tailored to perform a particular
evaluation task.

This page documents the feedback providers available in _TruLens_.

There are three categories of such providers as well as combination providers
that make use of one or more of these providers to offer additional feedback
functions based capabilities of the constituent providers.

## Classification-based Providers

Some feedback functions rely on classification typically tailor made for task, unlike LLM models.

- [Huggingface provider][trulens.providers.huggingface.Huggingface]
  containing a variety of classification-based feedback functions runnable on the remote Huggingface API.
  - [Huggingface Local provider][trulens.providers.huggingfacelocal.HuggingfaceLocal]
  containing a variety of classification-based feedback functions runnable locally.
- [OpenAI provider][trulens.providers.openai.OpenAI] (and
  subclasses) features moderation feedback functions.

## Generation-based Providers

Providers which use large language models for feedback evaluation:

- [OpenAI provider][trulens.providers.openai.OpenAI] or
  [AzureOpenAI provider][trulens.providers.openai.AzureOpenAI]
- [Bedrock provider][trulens.providers.bedrock.Bedrock]
- [LiteLLM provider][trulens.providers.litellm.LiteLLM]
- [_LangChain_ provider][trulens.providers.langchain.Langchain]

Feedback functions in common across these providers are in their abstract class
[LLMProvider][trulens.feedback.LLMProvider].

## Embedding-based Providers

- [Embeddings][trulens.feedback.embeddings.Embeddings]

## Provider Combinations

- [Groundtruth][trulens.feedback.groundtruth.GroundTruthAgreement]

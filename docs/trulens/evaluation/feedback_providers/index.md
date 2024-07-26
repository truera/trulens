# Feedback Providers

TruLens constructs feedback functions by combining more general models, known as
the [**_feedback provider_**][src.core.trulens.core.feedback.provider], and
[**_feedback implementation_**](../feedback_implementations/index.md) made up of
carefully constructed prompts and custom logic tailored to perform a particular
evaluation task.

This page documents the feedback providers available in _TruLens_.

There are three categories of such providers as well as combination providers
that make use of one or more of these providers to offer additional feedback
functions based capabilities of the constituent providers.

## Classification-based Providers

Some feedback functions rely on classification typically tailor made for task, unlike LLM models.

- [Huggingface provider][src.provider.huggingface.trulens.ext.provider.huggingface.provider.Huggingface]
  containing a variety of classification-based feedback functions runnable on the remote Huggingface API.
  - [Huggingface provider][src.provider.huggingface-local.trulens.ext.provider.huggingfacelocal.provider.Huggingface]
  containing a variety of classification-based feedback functions runnable locally.
- [OpenAI provider][src.provider.openai.trulens.ext.provider.openai.provider.OpenAI] (and
  subclasses) features moderation feedback functions.

## Generation-based Providers

Providers which use large language models for feedback evaluation:

- [OpenAI provider][src.provider.openai.trulens.ext.provider.openai.provider.OpenAI] or
  [AzureOpenAI provider][src.provider.openai.trulens.ext.provider.openai.provider.AzureOpenAI]
- [Bedrock provider][src.provider.bedrock.trulens.ext.provider.bedrock.provider.Bedrock]
- [LiteLLM provider][src.provider.litellm.trulens.ext.provider.litellm.provider.LiteLLM]
- [_LangChain_ provider][src.provider.langchain.trulens.ext.provider.langchain.provider.Langchain]

Feedback functions in common across these providers are in their abstract class
[LLMProvider][src.feedback.trulens.feedback.llm_provider.LLMProvider].

## Embedding-based Providers

- [Embeddings][src.feedback.trulens.feedback.embeddings.Embeddings]

## Provider Combinations

- [Groundtruth][src.feedback.trulens.feedback.groundtruth.GroundTruthAgreement]

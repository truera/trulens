# Feedback Providers

TruLens constructs feedback functions by combining more general models, known as the [**_feedback provider_**][trulens_eval.feedback.provider.base.Provider], and [**_feedback implementation_**](../feedback_implementations/index.md) made up of carefully constructed prompts and custom logic tailored to perform a particular evaluation task.

This page documents the feedback providers available in _TruLens_.

There are three categories of such providers as well as combination providers that make use of one or more of these providers to offer additional feedback functions based capabilities of the constituent providers.

## Classification-based Providers

Some feedback functions rely on classification typically tailor made for task, unlike LLM models.

- [Huggingface provider][trulens_eval.feedback.provider.hugs.Huggingface]
  containing a variety of feedback functions.
- [OpenAI provider][trulens_eval.feedback.provider.openai.OpenAI] (and
  subclasses) features moderation feedback functions.

## Generation-based Providers

Providers which use large language models for feedback evaluation:

- [OpenAI provider][trulens_eval.feedback.provider.openai.OpenAI] or
  [AzureOpenAI provider][trulens_eval.feedback.provider.openai.AzureOpenAI]
- [Bedrock provider][trulens_eval.feedback.provider.bedrock.Bedrock]
- [LiteLLM provider][trulens_eval.feedback.provider.litellm.LiteLLM]
- [_LangChain_ provider][trulens_eval.feedback.provider.langchain.Langchain]

Feedback functions in common across these providers are in their abstract class
[LLMProvider][trulens_eval.feedback.provider.base.LLMProvider].

## Embedding-based Providers

- [Embeddings][trulens_eval.feedback.embeddings.Embeddings]

## Provider Combinations

- [Groundedness][trulens_eval.feedback.groundedness.Groundedness]

- [Groundtruth][trulens_eval.feedback.groundtruth.GroundTruthAgreement]

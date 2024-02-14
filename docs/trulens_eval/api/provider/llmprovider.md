# LLM Provider

LLMProvider is the base class containing feedback functions that use an LLM for
evaluation. This includes:

* [OpenAI][trulens_eval.feedback.provider.openai.OpenAI] and subclass [AzureOpenAI][trulens_eval.feedback.provider.openai.AzureOpenAI].
* [Bedrock][trulens_eval.feedback.provider.bedrock.Bedrock].
* [LiteLLM][trulens_eval.feedback.provider.litellm.LiteLLM].
* [Langchain][trulens_eval.feedback.provider.langchain.Langchain].

::: trulens_eval.feedback.provider.base.LLMProvider

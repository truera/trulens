# Generation-based Providers

Providers which use large language models for feedback evaluation:

- [OpenAI provider][trulens_eval.feedback.provider.openai.OpenAI] or
  [AzureOpenAI provider][trulens_eval.feedback.provider.openai.AzureOpenAI]
- [Bedrock provider][trulens_eval.feedback.provider.bedrock.Bedrock]
- [LiteLLM provider][trulens_eval.feedback.provider.litellm.LiteLLM]
- [Langchain provider][trulens_eval.feedback.provider.langchain.Langchain]

Feedback functions in common across these providers are in their abstract class [LLMProvider][trulens_eval.feedback.provider.base.LLMProvider].
# 🦜️🔗 Langchain Provider

Below is how you can instantiate a [Langchain LLM](https://python.langchain.com/docs/modules/model_io/llms/) as a provider.

All feedback functions listed in the base [LLMProvider
class][trulens_eval.feedback.provider.base.LLMProvider] can be run with the Langchain Provider.

!!! note

    Langchain provider cannot be used in `deferred` mode due to inconsistent serialization capabilities of langchain apps.

::: trulens_eval.feedback.provider.langchain.Langchain

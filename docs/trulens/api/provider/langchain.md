# 🦜️🔗 _LangChain_ Provider

Below is how you can instantiate a [_LangChain_ LLM](https://python.langchain.com/docs/modules/model_io/llms/) as a provider.

All feedback functions listed in the base [LLMProvider
class][trulens_eval.feedback.provider.base.LLMProvider] can be run with the _LangChain_ Provider.

!!! note

    _LangChain_ provider cannot be used in `deferred` mode due to inconsistent serialization capabilities of _LangChain_ apps.

::: src.provider.langchain.trulens.provider.langchain.provider.Langchain

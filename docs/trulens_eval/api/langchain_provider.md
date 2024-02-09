# Langchain Provider APIs

Below is how you can instantiate a [Langchain LLM](https://python.langchain.com/docs/modules/model_io/llms/) as a provider.

All feedback functions listed in the base [`LLMProvider` class](https://trulens.org/trulens_eval/api/feedback/#trulens_eval.feedback.provider.base.LLMProvider) can be run with the Langchain Provider.

!!! note

    Langchain provider cannot be used in `deferred` mode because chains cannot be serialized.

::: trulens_eval.feedback.provider.langchain.Langchain

# API Reference

Welcome to the TruLens API Reference! Use the search and navigation to explore
the various modules and classes available in the TruLens library.

## Required and 📦 Optional packages

These packages are installed when installing the main `trulens` package.

- `trulens-core` installs [core][trulens.core].

- `trulens-feedback` installs [feedback][trulens.feedback].

- `trulens-dashboard` installs [dashboard][trulens.dashboard].

- `trulens_eval` installs [trulens_eval](trulens_eval), a temporary package for backwards compatibility.

Three categories of optional packages contain integrations with 3rd party app
types and providers:

- [Apps](apps/index.md) for instrumenting apps.

    - 📦 [TruChain][trulens.apps.langchain.TruChain] in package
        `trulens-apps-langchain` for instrumenting LangChain apps.

    - 📦 [TruLlama][trulens.apps.llamaindex.TruLlama] in package
        `trulens-app-trullama` for instrumenting LlamaIndex apps.

    - 📦 [TruRails][trulens.apps.nemo.TruRails] in package
        `trulens-app-nemo` for instrumenting NeMo Guardrails apps.

- [Providers](providers/index.md) for invoking various models or using them for feedback functions.

    - 📦 [Cortex][trulens.providers.cortex.provider.Cortex] in the package
        `trulens-providers-cortex` for using Snowflake Cortex models.

    - 📦 [Langchain][trulens.providers.langchain.provider.Langchain] in the package
        `trulens-providers-langchain` for using models via Langchain.

    - 📦 [Bedrock][trulens.providers.bedrock.provider.Bedrock] in the package
        `trulens-providers-bedrock` for using Amazon Bedrock models.

    - 📦 [Huggingface][trulens.providers.huggingface.provider.Huggingface] and
        [HuggingfaceLocal][trulens.providers.huggingface.provider.HuggingfaceLocal]
        in the package `trulens-providers-huggingface` for using Huggingface models.

    - 📦 [LiteLLM][trulens.providers.litellm.provider.LiteLLM] in the package
        `trulens-providers-litellm` for using models via LiteLLM.

    - 📦 [OpenAI][trulens.providers.openai.provider.OpenAI] and
        [AzureOpenAI][trulens.providers.openai.provider.AzureOpenAI] in the package
        `trulens-providers-openai` for using OpenAI models.

- [Connectors](connectors/index.md) for storing TruLens data.

    - 📦 [SnowflakeConnector][trulens.connectors.snowflake.connector.SnowflakeConnector]
      in package `trulens-connectors-snowlake` for connecting to Snowflake
      databases.

Other optional packages:

- 📦 [Benchmark][trulens.benchmark] in package `trulens-benchmark` for running
  benchmarks and meta evaluations.

## Private API

Module members which begin with an underscore `_` are private are should not be
used by code outside of _TruLens_.

Module members which begin but not end with double undescore `__` are class/module private
and should not be used outside of the defining module or class.

!!! Warning
    There is no deprecation period for the private API.

# API Reference

Welcome to the TruLens API Reference! Use the search and navigation to explore
the various modules and classes available in the TruLens library.

## Required and Optional packages

These packages are installed when installing the main `trulens` package.

- `trulens-core` installs [core][trulens.core].

- `trulens-feedback` installs [feedback][trulens.feedback].

- `trulens-dashboard` installs [dashboard][trulens.dashboard].

- `trulens_eval` installs [trulens_eval](trulens_eval), a temporary package for backwards compatibility.

Two categories of optional packages contain integrations with 3rd party app
types and providers:

- [Apps](apps/index.md).

    - [TruChain][trulens.apps.langchain.TruChain] in package
        `trulens-app-langchain`.

    - [TruLlama][trulens.apps.llamaindex.TruLlama] in package
        `trulens-app-trullama`.

    - [TruRails][trulens.apps.nemo.TruRails] in package
        `trulens-app-nemo`.

- [Providers](providers/index.md).

    - [Cortex][trulens.providers.cortex.provider.Cortex] in the package
        `trulens-providers-cortex`.

    - [Langchain][trulens.providers.langchain.provider.Langchain] in the package
        `trulens-providers-langchain`.

    - [Bedrock][trulens.providers.bedrock.provider.Bedrock] in the package
        `trulens-providers-bedrock`.

    - [Huggingface][trulens.providers.huggingface.provider.Huggingface] and
        [HuggingfaceLocal][trulens.providers.huggingface.provider.HuggingfaceLocal]
        in the package `trulens-providers-huggingface`.

    - [LiteLLM][trulens.providers.litellm.provider.LiteLLM] in the package
        `trulens-providers-litellm`.

    - [OpenAI][trulens.providers.openai.provider.OpenAI] and
        [AzureOpenAI][trulens.providers.openai.provider.AzureOpenAI] in the package
        `trulens-providers-openai`.

Two additional optional packages:

- `trulens-connectors-snowflake` for connecting to Snowflake databases.

- `trulens-benchmark` for running benchmarks or meta evaluations.

## Private API

Module members which begin with an underscore `_` are private are should not be
used by code outside of _TruLens_.

Module members which begin but not end with double undescore `__` are class/module private
and should not be used outside of the defining module or class.

!!! Warning
    There is no deprecation period for the private API.

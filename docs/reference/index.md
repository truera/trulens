# API Reference

TruLens features are organized into two public APIs: interactive and production. The
interactive API is ideal for initial development and inside jupyter notebooks;
it makes it most convenient to get started with _TruLens_. The development API
is instead suited for deployed code.

## Interactive API

The interactive API is contained in the `trulens-api` package and exposes
the principal features of each of the _TruLens_ modules. These are enumerated in
the `__all__` special variable in each applicable module.

- [api][trulens.api] -- Contains everything to get started including
  workspace manager `Tru` and the contents of the other Interactive API modules
  below:

- [dashboard][trulens.api.dashboard] -- Includes dashboard.

- [providers][trulens.api.providers] -- Includes all providers.

- [instrument][trulens.api.instrument] -- Includes all recorder
  types.

- [feedback][trulens.api.feedback] -- Includes feedback
  configuration enums, feedback constructors, and selectors.

Interactive API usage does not require the manual installation of optional
packages as that is done automatically.

## Production API

The production API spans all `trulens-*` packages with modules outside of `trulens.api`.

The use of the production API may require the installation of optional packages.

### Required and Optional packages

These packages are installed when installing the main `trulens` package.

- [core](trulens/core)
- [feedback](trulens/feedback)
- [dashboard](trulens/dashboard)

Two categories of optional packages contain integrations with 3rd party app types and providers:

- [Instrument](instrument/index.md)

       - [TruChain][trulens.instrument.langchain.TruChain]

       - [TruLlama][trulens.instrument.llamaindex.TruLlama]

       - [TruRails][trulens.instrument.nemo.TruRails]

- [Providers](providers/index.md)

       - [Cortex][trulens.providers.cortex.provider.Cortex]

       - [Langchain][trulens.providers.langchain.provider.Langchain]

       - [Bedrock][trulens.providers.bedrock.provider.Bedrock]

       - [HuggingFace][trulens.providers.huggingface.provider.Huggingface]

       - [LiteLLM][trulens.providers.litellm.provider.LiteLLM]

       - [OpenAI][trulens.providers.openai.provider.OpenAI]

## Private API

Module members which begin with an underscore `_` are private are should not be
used by code outside of _TruLens_.

Module members which begin with double undescore `__` are class/module private
and should not be used outside of the defining module or class.

!!! Warning
    There is no deprecation period for the private API.

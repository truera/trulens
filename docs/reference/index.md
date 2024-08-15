# API Reference

TruLens features are organized into two public APIs: _interactive_ and
_production_. The interactive API is ideal for initial development and inside
jupyter notebooks; it makes it most convenient to get started with _TruLens_.
The development API is instead suited for deployed code. The differences between
these APIs are:

- By default, interactive API automatically installs optional `trulens-*`
  packages as needed. The production API does not do this.

- If automatic installation is disabled, the interactive API prints out
  instructions for installing optional packages if they are not already
  installed. The production API does not do this.

- When attempting to import a non-existant name from an interactive API module,
  a message is offered that enumerates its contents and the packages those
  contents depend on. The production API does not do this.

- The interactive API is composed entirely of module `__init__.py` files that
  export aliases to elements of the production API which itself has no
  `__init__.py` files or exported aliases. Code written using the production
  API, therefore, has to import components by their full path.

- The interactive API follows a looser set of coding standards than the
  production API.

Both APIs follow a [deprecation policy](trulens/contributing/policies.md).

## Interactive API

The interactive API is contained in the `trulens-api` package, `trulens.api`
module and submodules which expose the principal _TruLens_ features. These are
enumerated in the `__all__` special variable in each `__init__.py` of the
modules below:

- [trulens.api](trulens/api.md) -- Contains everything to get started including
  workspace manager [Tru][trulens.core.tru.Tru] and the contents of the other
  Interactive API modules below:

- [trulens.api.dashboard](trulens/api/dashboard.md) -- Includes dashboard.

- [trulens.api.providers](trulens/api/providers.md) -- Includes all providers.

- [trulens.api.instrument](trulens/api/instrument.md) -- Includes all recorder
  types.

- [trulens.api.feedback](trulens.api.feedback) -- Includes feedback
  configuration enums, feedback constructors, and selectors.

Interactive API usage does not require the manual installation of optional
packages as that is done automatically.

## Production API

The production API spans all `trulens-*` packages with modules outside of
`trulens.api`. No such package includes exported aliases such as those in
`__init__.py` files hence full names of components need to be used. To find the
path to a component, one can print it:

  ```python
  from trulens.api import Tru
  print(Tru)
  ```

Shows:

  ```
  <class 'trulens.core.tru.Tru'>
  ```

The use of the production API may require the installation of optional packages.

### Required and Optional packages

These packages are installed when installing the main `trulens` package.

- [core](trulens/core)
- [feedback](trulens/feedback)
- [dashboard](trulens/dashboard)

Two categories of optional packages contain integrations with 3rd party app
types and providers:

- [Instrument](instrument/index.md). The interactive interface to these features
  is the [trulens.api.instrument][trulens.api.instrument] module.

  - [TruChain][trulens.instrument.langchain.TruChain] in package
    `trulens-instrument-langchain`.

  - [TruLlama][trulens.instrument.llamaindex.TruLlama] in package
    `trulens-instrument-trullama`.

  - [TruRails][trulens.instrument.nemo.TruRails] in package
    `trulens-instrument-nemo`.

- [Providers](providers/index.md). The interactive interface to these features
  is the [trulens.api.providers][trulens.api.providers] module.

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

## Private API

Module members which begin with an underscore `_` are private are should not be
used by code outside of _TruLens_.

Module members which begin but not end with double undescore `__` are class/module private
and should not be used outside of the defining module or class.

!!! Warning
    There is no deprecation period for the private API.

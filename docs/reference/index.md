# API Reference

TruLens features are organized into two public APIs: interactive and production. The
interactive API is ideal for initial development and inside jupyter notebooks;
it makes it most convenient to get started with _TruLens_. The development API
is instead suited for deployed code.

## Interactive API

The interactive API is contained in the `trulens-api` package and exposes
the principal features of each of the _TruLens_ modules. These are enumerated in
the `__all__` special variable in each applicable module.

- [trulens][trulens] -- Contains everything to get started including workspace manager `Tru` and the contents of the other Interactive API modules below:

::: trulens
    options:
        heading_level: 4
        show_bases: false
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_docstring_classes: false
        show_docstring_functions: false
        show_docstring_modules: false
        show_docstring_parameters: false
        show_docstring_returns: false
        show_docstring_description: false
        show_docstring_examples: false
        show_docstring_other_parameters: false
        show_docstring_attributes: false
        show_signature: false
        show_submodules: false
        separate_signature: false
        summary: false
        group_by_category: false
        members_order: alphabetical

- [trulens.providers][trulens.providers] -- Includes all providers.

::: trulens.providers
    options:
        heading_level: 4
        show_bases: false
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_docstring_classes: false
        show_docstring_functions: false
        show_docstring_modules: false
        show_docstring_parameters: false
        show_docstring_returns: false
        show_docstring_description: false
        show_docstring_examples: false
        show_docstring_other_parameters: false
        show_docstring_attributes: false
        show_inherited_members: false
        show_signature: false
        show_submodules: false
        separate_signature: false
        summary: false
        group_by_category: false
        members_order: alphabetical
        filters:
            - "!^[a-z]"

       - [Cortex][trulens.providers.cortex.provider.Cortex]

       - [Langchain][trulens.providers.langchain.provider.Langchain]

       - [Bedrock][trulens.providers.bedrock.provider.Bedrock]

       - [HuggingFace][trulens.providers.huggingface.provider.Huggingface]

       - [LiteLLM][trulens.providers.litellm.provider.LiteLLM]

       - [OpenAI][trulens.providers.openai.provider.OpenAI]

- [trulens.instrument][trulens.instrument] -- Includes all recorder types.

::: trulens.instrument
    options:
        heading_level: 4
        show_bases: false
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_docstring_classes: false
        show_docstring_functions: false
        show_docstring_modules: false
        show_docstring_parameters: false
        show_docstring_returns: false
        show_docstring_description: false
        show_docstring_examples: false
        show_docstring_other_parameters: false
        show_docstring_attributes: false
        show_signature: false
        show_submodules: false
        separate_signature: false
        summary: false
        group_by_category: false
        members_order: alphabetical
        show_inherited_members: false
        filters:
            - "!^[a-z]"

       - [TruChain][trulens.instrument.langchain.TruChain]

       - [TruLlama][trulens.instrument.llamaindex.TruLlama]

       - [TruRails][trulens.instrument.nemo.TruRails]

- [trulens.feedback][trulens.feedback] -- Includes feedback configuration enums, feedback constructors, and selectors.

::: trulens.feedback
    options:
        heading_level: 4
        show_bases: false
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_docstring_classes: true
        show_docstring_functions: true
        show_docstring_modules: true
        show_docstring_parameters: true
        show_docstring_returns: true
        show_docstring_description: true
        show_docstring_examples: true
        show_docstring_other_parameters: true
        show_docstring_attributes: true
        show_signature: false
        show_submodules: false
        separate_signature: false
        summary: true
        group_by_category: false
        members_order: alphabetical
        show_inherited_members: true


Interactive API usage does not require the manual installation of optional packages as that is done automatically.

## Production API

The production API spans all `trulens-*` packages and refers to public names
that are not enumerated in `__all__`.

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

## Deprecation Schedule

Changes to the public API are governed by a deprecation process in three stages.
In the warning period of no less than _3 months_, the use of a deprecated package, module, or value will produce a warning but otherwise operate as expected. In the subsequent deprecated period of no less than _3 months_, the use of that component will produce an error after the deprecation message. After these two periods, the deprecated capability will be completely removed.

!!! Info "Deprecation process"
    - 3 months - warning

    - 3 months - warning + error

    - removal

Changes that result in non-backwards compatible functionality are also reflected in the version numbering. In such cases, the appropriate level version change will occur at the introduction of the warning period.

### Currently derpecating features

- Starting 1.0.0, the `trulens_eval` package is being deprecated in favor of `trulens` and several associated required and optional packages. See [trulens_eval migration](/trulens/guides/trulens_eval_migration) for details.

    - Warning period: 2024-09-01 with `trulens-eval` 1.0.0a0 -- 2024-12-01. Backwards compatibility during the warning period is provided by the new content of the `trulens_eval` package which provides aliases to the features in their new locations. See [trulens_eval](trulens/api/trulens_eval/index.md).

    - Deprecated period: 2024-09-01 -- 2025-02-01 . Usage of `trulens_eval` will produce warnings and errors.

    - Removed expected 2024-02-01 Installation of the latest version of `trulens_eval` will be an error itself.

## Private API

Module members which begin with an underscore `_` are private are should not be
used by code outside of _TruLens_.

Module members which begin with double undescore `__` are class/module private
and should not be used outside of the defining module or class.

!!! Warning
    There is no deprecation period for the private API.

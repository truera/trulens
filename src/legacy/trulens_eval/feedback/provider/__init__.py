# ruff: noqa: E402
"""Aliases.

Aliases to new locations of various names. This module is deprecated and will be
removed in a future release.
"""

from trulens.core.utils import deprecation

from trulens_eval import packages_dep_warn

packages_dep_warn("trulens_eval.feedback.provider")

from trulens.core.feedback.provider import Provider
from trulens.core.utils import imports as imports_utils

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_LITELLM
):
    from trulens.providers.litellm.provider import LiteLLM

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_BEDROCK
):
    from trulens.providers.bedrock.provider import Bedrock

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_OPENAI
):
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_CORTEX
):
    from trulens.providers.cortex.provider import Cortex

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE
):
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_LANGCHAIN
):
    from trulens.providers.langchain.provider import Langchain

__all__ = [
    "Provider",
    "OpenAI",
    "AzureOpenAI",
    "Huggingface",
    "HuggingfaceLocal",
    "LiteLLM",
    "Bedrock",
    "Langchain",
    "Cortex",
]

# Replace all classes we expose to ones which issue a deprecation warning upon
# initialization.
deprecation.moved(
    globals(),
    names=__all__,
    old="trulens_eval.feedback.provider",
    new="trulens.providers",
)

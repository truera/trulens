# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.feedback` and
    `trulens.providers.*` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

from trulens_eval._utils import optional as optional_utils

deprecation_utils.packages_dep_warn()

from trulens.core.feedback.provider import Provider
from trulens.core.utils import imports as import_utils

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_LITELLM
):
    from trulens.providers.litellm.provider import LiteLLM

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_BEDROCK
):
    from trulens.providers.bedrock.provider import Bedrock

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_OPENAI
):
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_CORTEX
):
    from trulens.providers.cortex.provider import Cortex

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_HUGGINGFACE
):
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_LANGCHAIN
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
deprecation_utils.moved(
    globals(),
    names=__all__,
    old="trulens_eval.feedback.provider",
    new="trulens.providers",
)

# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.feedback` and
    `trulens.providers.*` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

from trulens_eval._utils import optional as optional_utils

deprecation_utils.packages_dep_warn()

from trulens.core.feedback.endpoint import Endpoint
from trulens.core.utils import imports as import_utils
from trulens.feedback.dummy.endpoint import DummyEndpoint

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_LITELLM
):
    from trulens.providers.litellm.endpoint import LiteLLMEndpoint

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_BEDROCK
):
    from trulens.providers.bedrock.endpoint import BedrockEndpoint

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_OPENAI
):
    from trulens.providers.openai.endpoint import OpenAIClient
    from trulens.providers.openai.endpoint import OpenAIEndpoint

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_CORTEX
):
    from trulens.providers.cortex.endpoint import CortexEndpoint

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_LANGCHAIN
):
    from trulens.providers.langchain.endpoint import LangchainEndpoint

with import_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_HUGGINGFACE
):
    from trulens.providers.huggingface.endpoint import HuggingfaceEndpoint

__all__ = [
    "Endpoint",
    "DummyEndpoint",
    "HuggingfaceEndpoint",
    "OpenAIEndpoint",
    "LiteLLMEndpoint",
    "BedrockEndpoint",
    "OpenAIClient",
    "LangchainEndpoint",
    "CortexEndpoint",
]

# Replace all classes we expose to ones which issue a deprecation warning upon
# initialization.
deprecation_utils.moved(
    globals(),
    names=__all__,
    old="trulens_eval.feedback.provider.endpoint",
    new="trulens.providers",
)

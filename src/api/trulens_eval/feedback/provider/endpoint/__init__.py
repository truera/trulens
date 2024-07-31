"""Aliases.

Aliases to new locations of various names. This module is deprecated and will be
removed in a future release.
"""

from trulens_eval import dep_warn
from trulens.core.utils import deprecation

dep_warn("trulens_eval.feedback.provider.endpoint")

from trulens.core.utils import imports as imports_utils

from trulens.core.feedback.endpoint import DummyEndpoint
from trulens.core.feedback.endpoint import Endpoint

with imports_utils.OptionalImports(
        messages=imports_utils.REQUIREMENT_PROVIDER_LITELLM):
    from trulens.providers.litellm.endpoint import LiteLLMEndpoint

with imports_utils.OptionalImports(
        messages=imports_utils.REQUIREMENT_PROVIDER_BEDROCK):
    from trulens.providers.bedrock.endpoint import BedrockEndpoint

with imports_utils.OptionalImports(
        messages=imports_utils.REQUIREMENT_PROVIDER_OPENAI):
    from trulens.providers.openai.endpoint import OpenAIClient
    from trulens.providers.openai.endpoint import OpenAIEndpoint

with imports_utils.OptionalImports(
        messages=imports_utils.REQUIREMENT_PROVIDER_CORTEX):
    from trulens.providers.cortex.endpoint import CortexEndpoint

with imports_utils.OptionalImports(
        messages=imports_utils.REQUIREMENT_PROVIDER_LANGCHAIN):
    from trulens.providers.langchain.endpoint import LangchainEndpoint

with imports_utils.OptionalImports(
        messages=imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE):
    from trulens.providers.huggingface.endpoint import HuggingfaceEndpoint

__all__ = [
    "Endpoint", "DummyEndpoint", "HuggingfaceEndpoint", "OpenAIEndpoint",
    "LiteLLMEndpoint", "BedrockEndpoint", "OpenAIClient", "LangchainEndpoint",
    "CortexEndpoint"
]

# Replace all classes we expose to ones which issue a deprecation warning upon
# initialization.
deprecation.moved(globals(),
                  names=__all__,
                  old="trulens_eval.feedback.provider.endpoint",
                  new="trulens.providers")

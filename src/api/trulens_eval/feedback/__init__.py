"""Aliases.

Aliases to new locations of various names. This module is deprecated and will be
removed in a future release.
"""

from trulens_eval import dep_warn
from trulens.core.utils import deprecation

dep_warn("trulens_eval.feedback")

from trulens.core.utils import imports as imports_utils

with imports_utils.OptionalImports(messages=imports_utils.REQUIREMENT_FEEDBACK):
    from trulens.feedback.feedback import Feedback
    from trulens.feedback.embeddings import Embeddings
    from trulens.feedback.groundtruth import GroundTruthAgreement

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
    messages=imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE
):
    from trulens.providers.huggingface.provider import Huggingface

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE_LOCAL
):
    from trulens.providers.huggingfacelocal.provider import HuggingfaceLocal

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_LANGCHAIN
):
    from trulens.providers.langchain.provider import Langchain

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_CORTEX
):
    from trulens.providers.cortex.provider import Cortex

__all__ = [
    "Feedback",
    "Embeddings",
    "GroundTruthAgreement",
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
    old="trulens_eval.feedback",
    new="trulens.feedback",
)

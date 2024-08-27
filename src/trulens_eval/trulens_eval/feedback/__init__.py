# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.feedback`
    or `trulens.feedback` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

from trulens_eval._utils import optional as optional_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils import imports as imports_utils

with imports_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_FEEDBACK
):
    from trulens.feedback.embeddings import Embeddings
    from trulens.feedback.feedback import Feedback
    from trulens.feedback.groundtruth import GroundTruthAgreement

with imports_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_LITELLM
):
    from trulens.providers.litellm.provider import LiteLLM

with imports_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_BEDROCK
):
    from trulens.providers.bedrock.provider import Bedrock

with imports_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_OPENAI
):
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

with imports_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_HUGGINGFACE
):
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal

with imports_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_LANGCHAIN
):
    from trulens.providers.langchain.provider import Langchain

with imports_utils.OptionalImports(
    messages=optional_utils.REQUIREMENT_PROVIDER_CORTEX
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
deprecation_utils.moved(
    globals(),
    names=__all__,
    old="trulens_eval.feedback",
    new="trulens.feedback",
)

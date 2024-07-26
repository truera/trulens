"""Aliases.

Aliases to new locations of various names. This module is deprecated and will be
removed in a future release.
"""

import warnings
warnings.warn(
    "The `trulens_eval.feedback` module is deprecated. "
    "Use `trulens.feedback` from the `trulens-feedback` package instead.",
    DeprecationWarning,
    stacklevel=2
)

# Specific feedback functions:
# Main class holding and running feedback functions:
from trulens.core.utils import imports as mod_imports_utils

with mod_imports_utils.OptionalImports(
    messages=mod_imports_utils.REQUIREMENT_FEEDBACK):
    from trulens.feedback.feedback import Feedback
    from trulens.feedback.embeddings import Embeddings
    from trulens.feedback.groundtruth import GroundTruthAgreement

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_LITELLM):
    from trulens.providers.litellm.provider import LiteLLM

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_BEDROCK):
    from trulens.providers.bedrock.provider import Bedrock

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_OPENAI):
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE):
    from trulens.providers.huggingface.provider import Huggingface

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE_LOCAL):
    from trulens.providers.huggingfacelocal.provider import HuggingfaceLocal

with mod_imports_utils.OptionalImports(
    messages=mod_imports_utils.REQUIREMENT_PROVIDER_LANGCHAIN):
    from trulens.providers.langchain.provider import Langchain

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_CORTEX):
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
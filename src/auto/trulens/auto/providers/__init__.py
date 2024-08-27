# ruff: noqa: E402
"""TruLens providers and provider constructors."""

from typing import TYPE_CHECKING

from trulens.auto._utils import auto as auto_utils

if TYPE_CHECKING:
    # Needed for static tools:
    from trulens.core.feedback.provider import Provider
    from trulens.feedback.dummy.provider import DummyProvider
    from trulens.feedback.embeddings import Embeddings
    from trulens.feedback.groundtruth import GroundTruthAggregator
    from trulens.feedback.groundtruth import GroundTruthAgreement
    from trulens.feedback.llm_provider import LLMProvider
    from trulens.providers.bedrock.provider import Bedrock
    from trulens.providers.cortex.provider import Cortex
    from trulens.providers.huggingface.provider import Dymmy as HuggingfaceDummy
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal
    from trulens.providers.langchain.provider import Langchain
    from trulens.providers.litellm.provider import LiteLLM
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

# Providers:
_PROVIDERS = {
    "DummyProvider": ("trulens-feedback", "trulens.feedback.dummy.provider"),
    "Bedrock": (
        "trulens-providers-bedrock",
        "trulens.providers.bedrock.provider",
    ),
    "Cortex": ("trulens-providers-cortex", "trulens.providers.cortex.provider"),
    "Huggingface": (
        "trulens-providers-huggingface",
        "trulens.providers.huggingface.provider",
    ),
    "HuggingfaceLocal": (
        "trulens-providers-huggingface",
        "trulens.providers.huggingface.provider",
    ),
    "HuggingfaceDummy": (
        "trulens-providers-huggingface",
        "trulens.providers.huggingface.provider",
        "Dummy",
    ),
    "Langchain": (
        "trulens-providers-langchain",
        "trulens.providers.langchain.provider",
    ),
    "LiteLLM": (
        "trulens-providers-litellm",
        "trulens.providers.litellm.provider",
    ),
    "OpenAI": ("trulens-providers-openai", "trulens.providers.openai.provider"),
    "AzureOpenAI": (
        "trulens-providers-openai",
        "trulens.providers.openai.provider",
    ),
}

_CONSTRUCTORS = {
    "Embeddings": ("trulens-feedback", "trulens.feedback.embeddings"),
    "GroundTruthAggregator": (
        "trulens-feedback",
        "trulens.feedback.groundtruth",
    ),
    "GroundTruthAgreement": (
        "trulens-feedback",
        "trulens.feedback.llm_provider",
    ),
}

_INTERFACES = {
    "Provider": ("trulens-core", "trulens.core.feedback.provider"),
    "LLMProvider": ("trulens-feedback", "trulens.feedback.llm_provider"),
}

_KINDS = {
    "provider": _PROVIDERS,
    "constructor": _CONSTRUCTORS,
    "interface": _INTERFACES,
}

__getattr__ = auto_utils.make_getattr_override(kinds=_KINDS)

__all__ = [
    # providers
    "DummyProvider",
    "Bedrock",
    "Cortex",
    "Huggingface",
    "HuggingfaceLocal",
    "HuggingfaceDummy",
    "Langchain",
    "LiteLLM",
    "OpenAI",
    "AzureOpenAI",
    # constructors
    "Embeddings",
    "GroundTruthAggregator",
    "GroundTruthAgreement",
    # interfaces
    "Provider",
    "LLMProvider",
]

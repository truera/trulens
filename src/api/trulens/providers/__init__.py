# ruff: noqa: E402
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from typing import TYPE_CHECKING

from trulens.core.utils import imports as import_utils

if TYPE_CHECKING:
    from trulens.providers.bedrock.provider import Bedrock
    from trulens.providers.cortex.provider import Cortex
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal
    from trulens.providers.langchain.provider import Langchain
    from trulens.providers.litellm.provider import LiteLLM
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

# Providers:
_PROVIDERS = {
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

_KINDS = {"provider": _PROVIDERS}

help, help_str = import_utils.make_help_str(_KINDS)

__getattr__ = import_utils.make_getattr_override(_KINDS, help_str)

# This has to be statically assigned though we would prefer to use _OPTIONAL_PROVIDERS.keys():
__all__ = [
    "Bedrock",
    "Cortex",
    "Huggingface",
    "HuggingfaceLocal",
    "Langchain",
    "LiteLLM",
    "OpenAI",
    "AzureOpenAI",
]

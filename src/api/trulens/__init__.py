# ruff: noqa: E402, F822

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# TODO: get this from poetry
__version_info__ = (1, 0, 0, "a0")
"""Version number components for major, minor, patch."""

__version__ = ".".join(map(str, __version_info__))
"""Version number string."""

from typing import TYPE_CHECKING

from trulens.core.utils import imports as import_utils


def set_no_install(val: bool = True) -> None:
    """Sets the NO_INSTALL flag to make sure optional packages are not
    automatically installed."""

    import_utils.NO_INSTALL = val


if TYPE_CHECKING:
    # Common classes:
    from trulens.core.feedback.feedback import Feedback
    from trulens.core.feedback.provider import Provider

    # TODO: other enums
    # Utilities
    from trulens.core.schema import Select

    # Enums
    from trulens.core.schema.feedback import FeedbackMode
    from trulens.core.tru import Tru
    from trulens.core.utils.threading import TP

_CLASSES = {
    "Tru": ("trulens-core", "trulens.core.tru"),
    "TP": ("trulens-core", "trulens.core.utils.threading"),
    "Feedback": ("trulens-core", "trulens.core.feedback.feedback"),
    "Provider": ("trulens-core", "trulens.core.feedback.provider"),
}

_ENUMS = {
    "FeedbackMode": ("trulens-core", "trulens.core.schema.feedback"),
}
_UTILITIES = {
    "Select": ("trulens-core", "trulens.core.schema"),
}

# Providers:

if TYPE_CHECKING:
    from trulens.providers.bedrock.provider import Bedrock
    from trulens.providers.cortex.provider import Cortex
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal
    from trulens.providers.langchain.provider import Langchain
    from trulens.providers.litellm.provider import LiteLLM
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

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

# Recorders:
if TYPE_CHECKING:
    from trulens.core.app.basic import TruBasicApp
    from trulens.core.app.custom import TruCustomApp
    from trulens.core.app.virtual import TruVirtual
    from trulens.instrument.langhain.tru_chain import TruChain
    from trulens.instrument.llamaindex.tru_llama import TruLlama
    from trulens.instrument.nemo.tru_rails import TruRails

_RECORDERS = {
    "TruBasicApp": ("trulens-core", "trulens.core.app.tru_basic_app"),
    "TruCustomApp": ("trulens-core", "trulens.core.app.tru_custom_app"),
    "TruVirtual": ("trulens-core", "trulens.core.app.tru_virtual"),
    "TruChain": (
        "trulens-instrument-langchain",
        "trulens.instrument.langchain.tru_chain",
    ),
    "TruLlama": (
        "trulens-instrument-llamaindex",
        "trulens.instrument.llamaindex.tru_llama",
    ),
    "TruRails": (
        "trulens-instrument-nemo",
        "trulens.instrument.nemo.tru_rails",
    ),
}

_KINDS = {
    "provider": _PROVIDERS,
    "recorder": _RECORDERS,
    "class": _CLASSES,
    "enum": _ENUMS,
    "utility": _UTILITIES,
}

help, help_str = import_utils.make_help_str(_KINDS)

__getattr__ = import_utils.make_getattr_override(_KINDS, help_str=help_str)

__all__ = [
    "Tru",  # main interface
    # app types
    "TruBasicApp",
    "TruCustomApp",
    "TruVirtual",
    # Cannot use dynamic: *list(_OPTIONAL_APPS.keys()),
    "TruChain",
    "TruLlama",
    "TruRails",
    # app setup
    "FeedbackMode",
    # feedback setup
    "Feedback",
    "Select",
    # feedback providers
    "Provider",
    # Cannot use dynamic: *list(_OPTIONAL_PROVIDERS.keys()),
    "AzureOpenAI",
    "OpenAI",
    "Langchain",
    "LiteLLM",
    "Bedrock",
    "Huggingface",
    "HuggingfaceLocal",
    "Cortex",
    # misc utility
    "TP",
]

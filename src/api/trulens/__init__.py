# ruff: noqa: E402, F822

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# TODO: get this from poetry
__version_info__ = (1, 0, 0, "a0")
"""Version number components for major, minor, patch."""

__version__ = ".".join(map(str, __version_info__))
"""Version number string."""

# This check is intentionally done ahead of the other imports as we want to
# print out a nice warning/error before an import error happens further down
# this sequence.
# from trulens.utils.imports import check_imports

from typing import TYPE_CHECKING

from trulens.core import schema as core_schema
from trulens.core import tru as mod_tru
from trulens.core.app import basic as mod_tru_basic_app
from trulens.core.app import custom as mod_tru_custom_app
from trulens.core.app import virtual as mod_tru_virtual
from trulens.core.feedback import feedback as mod_feedback
from trulens.core.feedback import provider as mod_provider
from trulens.core.schema import feedback as feedback_schema
from trulens.core.utils import imports as import_utils
from trulens.core.utils import threading as threading_utils
from trulens.core.utils.imports import get_package_version


def set_no_install(val: bool = True) -> None:
    """Sets the NO_INSTALL flag to make sure optional packages are not
    automatically installed."""

    import_utils.NO_INSTALL = val


# Common classes:
Tru = mod_tru.Tru

TP = threading_utils.TP
Feedback = mod_feedback.Feedback
Provider = mod_provider.Provider
FeedbackMode = feedback_schema.FeedbackMode
# TODO: other enums
Select = core_schema.Select

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


# Recorders:
TruBasicApp = mod_tru_basic_app.TruBasicApp
TruCustomApp = mod_tru_custom_app.TruCustomApp
TruVirtual = mod_tru_virtual.TruVirtual

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

if TYPE_CHECKING:
    from trulens.instrument.langhain.tru_chain import TruChain
    from trulens.instrument.llamaindex.tru_llama import TruLlama
    from trulens.instrument.nemo.tru_rails import TruRails
    from trulens.providers.bedrock.provider import Bedrock
    from trulens.providers.cortex.provider import Cortex
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal
    from trulens.providers.langchain.provider import Langchain
    from trulens.providers.litellm.provider import LiteLLM
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

_KINDS = {
    "provider": _PROVIDERS,
    "recorder": _RECORDERS,
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
    "get_package_version",
]

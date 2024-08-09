# ruff: noqa: E402, F822

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# TODO: get this from poetry
__version_info__ = (1, 0, 0, "a0")
"""Version number components for major, minor, patch."""

__version__ = ".".join(map(str, __version_info__))
"""Version number string."""

from typing import TYPE_CHECKING

from trulens import feedback as mod_feedback
from trulens import instrument as mod_instrument
from trulens import providers as mod_providers
from trulens.core.utils import imports as import_utils


def set_no_install(val: bool = True) -> None:
    """Sets the NO_INSTALL flag to make sure optional packages are not
    automatically installed."""

    import_utils.NO_INSTALL = val


if TYPE_CHECKING:
    # Common classes:
    from trulens.core.feedback.feedback import Feedback
    from trulens.core.feedback.provider import Provider

    # schema enums
    from trulens.core.schema import feedback as feedback_schema
    from trulens.core.tru import Tru
    from trulens.core.utils.threading import TP

    FeedbackMode = feedback_schema.FeedbackMode
    FeedbackResultStatus = feedback_schema.FeedbackResultStatus
    FeedbackOnMissingParameters = feedback_schema.FeedbackOnMissingParameters
    FeedbackCombinations = feedback_schema.FeedbackCombinations

    # schema classes
    FeedbackResult = feedback_schema.FeedbackResult
    FeedbackCall = feedback_schema.FeedbackCall
    FeedbackDefinition = feedback_schema.FeedbackDefinition

    # Utilities
    from trulens.core.schema import Select


_CLASSES = {  # **mod_feedback._CLASSES, # these are too specific to be included here
    "Tru": ("trulens-core", "trulens.core.tru"),
    "TP": ("trulens-core", "trulens.core.utils.threading"),
    "Feedback": ("trulens-core", "trulens.core.feedback.feedback"),
    "Provider": ("trulens-core", "trulens.core.feedback.provider"),
}

_ENUMS = mod_feedback._ENUMS

_UTILITIES = mod_feedback._UTILITIES

_SCHEMAS = mod_feedback._SCHEMAS

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

_PROVIDERS = mod_providers._PROVIDERS

# Recorders:
if TYPE_CHECKING:
    from trulens.core.app.basic import TruBasicApp
    from trulens.core.app.custom import TruCustomApp
    from trulens.core.app.virtual import TruVirtual
    from trulens.instrument.langhain.tru_chain import TruChain
    from trulens.instrument.llamaindex.tru_llama import TruLlama
    from trulens.instrument.nemo.tru_rails import TruRails

_RECORDERS = mod_instrument._RECORDERS

_KINDS = {
    "provider": _PROVIDERS,
    "recorder": _RECORDERS,
    "class": _CLASSES,
    "enum": _ENUMS,
    "utility": _UTILITIES,
    "schema": _SCHEMAS,
}

help, help_str = import_utils.make_help_str(_KINDS)

__getattr__ = import_utils.make_getattr_override(_KINDS, help_str=help_str)

__all__ = [
    # recorders types
    "TruBasicApp",
    "TruCustomApp",
    "TruVirtual",
    "TruChain",
    "TruLlama",
    "TruRails",
    # enums
    "FeedbackMode",
    "FeedbackResultStatus",
    "FeedbackOnMissingParameters",
    "FeedbackCombinations",
    # schema classes
    "FeedbackResult",
    "FeedbackCall",
    "FeedbackDefinition",
    # selector utilities
    "Select",
    # classes
    "Feedback",
    "Provider",
    "Tru",
    # providers
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

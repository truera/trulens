# ruff: noqa: E402, F822

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This is needed for static tools like mypy and mkdocstrings to figure out
    # content of this module.

    # Common classes:
    from trulens.core.feedback.feedback import Feedback
    from trulens.core.feedback.provider import Provider

    # Enums:
    from trulens.core.schema import feedback as feedback_schema

    # Schema enums:
    from trulens.core.tru import TruSession
    from trulens.core.utils.threading import TP

    FeedbackMode = feedback_schema.FeedbackMode
    FeedbackResultStatus = feedback_schema.FeedbackResultStatus
    FeedbackOnMissingParameters = feedback_schema.FeedbackOnMissingParameters
    FeedbackCombinations = feedback_schema.FeedbackCombinations

    # Dashboard utils
    from trulens.dashboard.run import run_dashboard
    from trulens.dashboard.run import stop_dashboard

    # Schema classes:
    FeedbackResult = feedback_schema.FeedbackResult
    FeedbackCall = feedback_schema.FeedbackCall
    FeedbackDefinition = feedback_schema.FeedbackDefinition

    # Recorders:
    from trulens.core.app.basic import TruBasicApp
    from trulens.core.app.custom import TruCustomApp
    from trulens.core.app.virtual import TruVirtual
    from trulens.core.schema import Select
    from trulens.instrument.langchain.tru_chain import TruChain
    from trulens.instrument.llamaindex.tru_llama import TruLlama
    from trulens.instrument.nemo.tru_rails import TruRails

    # Providers:
    from trulens.providers.bedrock.provider import Bedrock
    from trulens.providers.cortex.provider import Cortex
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal
    from trulens.providers.langchain.provider import Langchain
    from trulens.providers.litellm.provider import LiteLLM
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

_CLASSES = {
    # **mod_feedback._CLASSES, # these are too specific to be included here
    "TruSession": ("trulens-core", "trulens.core.session"),
    "TP": ("trulens-core", "trulens.core.utils.threading"),
    "Feedback": ("trulens-core", "trulens.core.feedback.feedback"),
    "Provider": ("trulens-core", "trulens.core.feedback.provider"),
}

from trulens.auto import dashboard as mod_dashboard
from trulens.auto import feedback as mod_feedback
from trulens.auto import instrument as mod_instrument
from trulens.auto import providers as mod_providers
from trulens.auto._utils import auto as auto_utils


def set_no_install(val: bool = True) -> None:
    """Sets the NO_INSTALL flag to make sure optional packages are not
    automatically installed."""
    auto_utils.NO_INSTALL = val


_UTILITIES = {
    **mod_feedback._UTILITIES,
    "set_no_install": ("trulens-auto", "trulens.auto"),
    "__version__": ("trulens-auto", "trulens.auto"),
}

_KINDS = {
    "provider": mod_providers._PROVIDERS,
    "recorder": mod_instrument._RECORDERS,
    "class": _CLASSES,
    "enum": mod_feedback._ENUMS,
    "utility": _UTILITIES,
    "schema": mod_feedback._SCHEMAS,
    "dashboard": mod_dashboard._FUNCTIONS,
}

help, help_str = auto_utils.make_help_str(_KINDS)

__getattr__ = auto_utils.make_getattr_override(_KINDS, help_str=help_str)

__all__ = [
    # recorders types
    "TruBasicApp",
    "TruCustomApp",
    "TruVirtual",
    "TruChain",
    "TruLlama",
    "TruRails",
    # dashboard utils,
    "run_dashboard",
    "stop_dashboard",
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
    "TruSession",
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
    "set_no_install",
    "__version__",
]

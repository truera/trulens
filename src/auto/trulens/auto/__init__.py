# ruff: noqa: E402, F822
"""TruLens notebook utilities."""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This is needed for static tools like mypy and mkdocstrings to figure out
    # content of this module.

    # Feedback specification schemas/classes/enums
    ## Main class
    from trulens.core.feedback.feedback import Feedback
    from trulens.core.schema import feedback as feedback_schema

    ## Schemas
    FeedbackResult = feedback_schema.FeedbackResult
    FeedbackCall = feedback_schema.FeedbackCall
    FeedbackDefinition = feedback_schema.FeedbackDefinition

    ## Enums
    FeedbackMode = feedback_schema.FeedbackMode
    FeedbackResultStatus = feedback_schema.FeedbackResultStatus
    FeedbackOnMissingParameters = feedback_schema.FeedbackOnMissingParameters
    FeedbackCombinations = feedback_schema.FeedbackCombinations

    # Sessions
    from trulens.connectors.snowflake.connector import SnowflakeDBConnector

    # Recorders:
    from trulens.core.app.basic import TruBasicApp
    from trulens.core.app.custom import TruCustomApp
    from trulens.core.app.virtual import TruVirtual

    # Connectors
    from trulens.core.database.connector.base import DefaultDBConnector
    from trulens.core.schema import Select
    from trulens.core.tru import TruSession

    # Dashboard
    from trulens.dashboard.run import run_dashboard
    from trulens.dashboard.run import stop_dashboard
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

_SESSION = {
    "TruSession": ("trulens-core", "trulens.core.session"),
}

from trulens.auto import connectors as mod_connectors
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
    "set_no_install": ("trulens-auto", "trulens.auto"),
    "__version__": ("trulens-auto", "trulens.auto"),
}

_KINDS = {
    "recorder": mod_instrument._RECORDERS,
    "provider": {**mod_providers._PROVIDERS, **mod_providers._CONSTRUCTORS},
    "feedback": {**mod_feedback._SPECS, **mod_feedback._CONFIGS},
    "dashboard": mod_dashboard._FUNCTIONS,
    "configuration": {**_SESSION, **_UTILITIES},
    "connector": mod_connectors._CONNECTORS,
}

__getattr__ = auto_utils.make_getattr_override(
    kinds=_KINDS,
    kinds_docs={
        "provider": "Providers are also available in `trulens.auto.providers`.",
        "recorder": "Recorders are also available in `trulens.auto.instrument`.",
        "dashboard": "Dashboard functions are also available in `trulens.auto.dashboard`.",
        "connector": "Connectors are also available in `trulens.auto.connectors`.",
        "feedback": "Feedback definition/configuration/result classes are also available in `trulens.auto.feedback`.",
    },
)

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
    # feedback specification enums
    "FeedbackMode",
    "FeedbackResultStatus",
    "FeedbackOnMissingParameters",
    "FeedbackCombinations",
    # feedback schemas
    "Feedback",
    "FeedbackResult",
    "FeedbackCall",
    "FeedbackDefinition",
    # feedback specification utilities
    "Select",
    # session
    "TruSession",
    # connectors
    "DefaultDBConnector",
    "SnowflakeDBConnector",
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
    "set_no_install",
    "__version__",
]

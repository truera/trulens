# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core`,
    `trulens.feedback`, `trulens.dashboard` instead.
"""

from typing import TYPE_CHECKING, Any
import warnings

from trulens.core.utils.deprecation import packages_dep_warn

packages_dep_warn("trulens_eval")

import trulens.api as api

if TYPE_CHECKING:
    from trulens.api import TP
    from trulens.api import AzureOpenAI
    from trulens.api import Bedrock
    from trulens.api import Cortex
    from trulens.api import Feedback
    from trulens.api import FeedbackMode
    from trulens.api import Huggingface
    from trulens.api import HuggingfaceLocal
    from trulens.api import Langchain
    from trulens.api import LiteLLM
    from trulens.api import OpenAI
    from trulens.api import Provider
    from trulens.api import Select
    from trulens.api import Tru
    from trulens.api import TruBasicApp
    from trulens.api import TruChain
    from trulens.api import TruCustomApp
    from trulens.api import TruLlama
    from trulens.api import TruRails
    from trulens.api import TruVirtual


def __getattr__(attr: str) -> Any:
    # Lazily get everything from trulens package.

    if attr in __all__:
        warnings.warn(
            f"Importing from `{__name__}` is deprecated. Use `trulens` instead:\nfrom trulens import {attr}.",
            DeprecationWarning,
            stacklevel=3,
        )

        return getattr(api, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


__all__ = [
    "Tru",  # main interface
    # app types
    "TruBasicApp",
    "TruCustomApp",
    "TruChain",
    "TruLlama",
    "TruVirtual",
    "TruRails",
    # app setup
    "FeedbackMode",
    # feedback setup
    "Feedback",
    "Select",
    # feedback providers
    "Provider",
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


# Replace all classes we expose to ones which issue a deprecation warning upon
# initialization.
# deprecation.moved(globals(), names=__all__, old="trulens_eval", new="trulens")

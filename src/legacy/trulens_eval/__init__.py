# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens` instead.
"""

# Must use this format due to griffe: https://github.com/mkdocstrings/griffe/commit/efba0c6a5e1dc185e96e5a09c05e94c751abc4cb
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from typing import TYPE_CHECKING, Any
import warnings

from trulens.core.utils.deprecation import packages_dep_warn

packages_dep_warn("trulens_eval")

import trulens

if TYPE_CHECKING:
    from trulens import TP
    from trulens import AzureOpenAI
    from trulens import Bedrock
    from trulens import Cortex
    from trulens import Feedback
    from trulens import FeedbackMode
    from trulens import Huggingface
    from trulens import HuggingfaceLocal
    from trulens import Langchain
    from trulens import LiteLLM
    from trulens import OpenAI
    from trulens import Provider
    from trulens import Select
    from trulens import Tru
    from trulens import TruBasicApp
    from trulens import TruChain
    from trulens import TruCustomApp
    from trulens import TruLlama
    from trulens import TruRails
    from trulens import TruVirtual


def __getattr__(attr: str) -> Any:
    # Lazily get everything from trulens package.

    if attr in __all__:
        warnings.warn(
            f"Importing from `{__name__}` is deprecated. Use `trulens` instead:\nfrom trulens import {attr}.",
            DeprecationWarning,
            stacklevel=3,
        )

        return getattr(trulens, attr)

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

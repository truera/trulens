# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core`
    or `trulens` instead.
"""

# Must use this format due to griffe: https://github.com/mkdocstrings/griffe/commit/efba0c6a5e1dc185e96e5a09c05e94c751abc4cb
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from trulens.core.utils.deprecation import packages_dep_warn

packages_dep_warn("trulens_eval")

# TODO: get this from poetry
__version_info__ = (1, 0, 0)
"""Version number components for major, minor, patch."""

__version__ = ".".join(map(str, __version_info__))
"""Version number string."""

# This check is intentionally done ahead of the other imports as we want to
# print out a nice warning/error before an import error happens further down
# this sequence.
# from trulens.utils.imports import check_imports

# check_imports()

from trulens.core import schema as core_schema
from trulens.core import tru as mod_tru
from trulens.core.app import basic as mod_tru_basic_app
from trulens.core.app import custom as mod_tru_custom_app
from trulens.core.app import virtual as mod_tru_virtual
from trulens.core.feedback import feedback as mod_feedback
from trulens.core.feedback import provider as mod_provider
from trulens.core.schema import feedback as feedback_schema
from trulens.core.utils import deprecation
from trulens.core.utils import imports as imports_utils
from trulens.core.utils import threading as threading_utils

# Optional provider types.

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_LITELLM
):
    from trulens.providers.litellm.provider import LiteLLM

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_BEDROCK
):
    from trulens.providers.bedrock.provider import Bedrock

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_OPENAI
):
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE
):
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_LANGCHAIN
):
    from trulens.providers.langchain.provider import Langchain

# the dependency snowflake-snowpark-python not yet supported in 3.12
with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_CORTEX
):
    from trulens.providers.cortex.provider import Cortex

# Optional app types.
with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_INSTRUMENT_LANGCHAIN
):
    from trulens.instrument.langchain.tru_chain import TruChain

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_INSTRUMENT_LLAMA
):
    from trulens.instrument.llamaindex.tru_llama import TruLlama

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_INSTRUMENT_NEMO
):
    from trulens.instrument.nemo.tru_rails import TruRails


Tru = mod_tru.Tru
TruBasicApp = mod_tru_basic_app.TruBasicApp
TruCustomApp = mod_tru_custom_app.TruCustomApp
TruVirtual = mod_tru_virtual.TruVirtual
TP = threading_utils.TP
Feedback = mod_feedback.Feedback
Provider = mod_provider.Provider
FeedbackMode = feedback_schema.FeedbackMode
Select = core_schema.Select

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
deprecation.moved(globals(), names=__all__, old="trulens_eval", new="trulens")

"""Aliases.

Aliases to new locations of various names. This module is deprecated and will be
removed in a future release.
"""

import warnings
warnings.warn(
    "The `trulens_eval` module is deprecated. "
    "See TODOLINK on migrating to the `trulens` packages.",
    DeprecationWarning,
    stacklevel=2
)

# TODO: get this from poetry
__version_info__ = (0, 33, 0)
"""Version number components for major, minor, patch."""

__version__ = '.'.join(map(str, __version_info__))
"""Version number string."""

# This check is intentionally done ahead of the other imports as we want to
# print out a nice warning/error before an import error happens further down
# this sequence.
# from trulens.utils.imports import check_imports

# check_imports()

from trulens.core import tru as mod_tru
from trulens.core.app import basic as mod_tru_basic_app
from trulens.core.app import custom as mod_tru_custom_app
from trulens.core.app import virtual as mod_tru_virtual
from trulens.core.feedback import feedback as mod_feedback
from trulens.core.feedback import provider as mod_provider
from trulens.core.schema import feedback as mod_feedback_schema
from trulens.core.utils import imports as mod_imports_utils
from trulens.core.utils import threading as mod_threading_utils

# Optional provider types.

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_LITELLM):
    from trulens.providers.litellm.provider import LiteLLM

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_BEDROCK):
    from trulens.providers.bedrock.provider import Bedrock

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_OPENAI):
    from trulens.providers.openai.provider import AzureOpenAI
    from trulens.providers.openai.provider import OpenAI

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE):
    from trulens.providers.huggingface.provider import Huggingface

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE_LOCAL):
    from trulens.providers.huggingfacelocal.provider import HuggingfaceLocal

with mod_imports_utils.OptionalImports(
    messages=mod_imports_utils.REQUIREMENT_PROVIDER_LANGCHAIN):
    from trulens.providers.langchain.provider import Langchain

# the dependency snowflake-snowpark-python not yet supported in 3.12
with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_PROVIDER_CORTEX):
    from trulens.providers.cortex.provider import Cortex

# Optional app types.
with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_INSTRUMENT_LANGCHAIN):
    from trulens.instrument.langchain.tru_chain import TruChain

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_INSTRUMENT_LLAMA):
    from trulens.instrument.llama.tru_llama import TruLlama

with mod_imports_utils.OptionalImports(
        messages=mod_imports_utils.REQUIREMENT_INSTRUMENT_NEMO):
    from trulens.instrument.nemo.tru_rails import TruRails


Tru = mod_tru.Tru
TruBasicApp = mod_tru_basic_app.TruBasicApp
TruCustomApp = mod_tru_custom_app.TruCustomApp
TruVirtual = mod_tru_virtual.TruVirtual
TP = mod_threading_utils.TP
Feedback = mod_feedback.Feedback
Provider = mod_provider.Provider
FeedbackMode = mod_feedback_schema.FeedbackMode
# Select = mod_feedback_schema.Select

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
#    "Select",

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
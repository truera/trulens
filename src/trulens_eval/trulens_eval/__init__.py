# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core`,
    `trulens.feedback`, `trulens.dashboard` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import imports as imports_utils

deprecation_utils.packages_dep_warn("trulens_eval")

from trulens.core.app.basic import TruBasicApp
from trulens.core.app.custom import TruCustomApp
from trulens.core.app.virtual import TruVirtual
from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.provider import Provider
from trulens.core.schema import Select
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.tru import Tru
from trulens.core.utils.threading import TP

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
    messages=imports_utils.REQUIREMENT_PROVIDER_CORTEX
):
    from trulens.providers.cortex.provider import Cortex

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_HUGGINGFACE
):
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingface.provider import HuggingfaceLocal

with imports_utils.OptionalImports(
    messages=imports_utils.REQUIREMENT_PROVIDER_LANGCHAIN
):
    from trulens.providers.langchain.provider import Langchain

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

__all__ = [
    # main interface
    "Tru",
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
deprecation_utils.moved(globals(), names=__all__)

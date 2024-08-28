# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.instrument.langchain.tru_chain` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.instrument.langchain.tru_chain import LangChainInstrument
from trulens.instrument.langchain.tru_chain import TruChain

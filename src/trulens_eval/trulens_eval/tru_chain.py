# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.apps.langchain.tru_chain` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.apps.langchain.tru_chain import LangChainInstrument
from trulens.apps.langchain.tru_chain import TruChain

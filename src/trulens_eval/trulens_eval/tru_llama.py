# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.instrument.llamaindex.tru_llama` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.instrument.llamaindex.tru_llama import LlamaInstrument
from trulens.instrument.llamaindex.tru_llama import TruLlama
from trulens.instrument.llamaindex.tru_llama import legacy

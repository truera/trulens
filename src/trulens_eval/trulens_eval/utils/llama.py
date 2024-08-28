# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.instrument.llamaindex.llama` instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.instrument.llamaindex.llama import COMPONENT_VIEWS
from trulens.instrument.llamaindex.llama import LLM
from trulens.instrument.llamaindex.llama import Agent
from trulens.instrument.llamaindex.llama import Other
from trulens.instrument.llamaindex.llama import Prompt
from trulens.instrument.llamaindex.llama import Tool
from trulens.instrument.llamaindex.llama import component_of_json
from trulens.instrument.llamaindex.llama import constructor_of_class

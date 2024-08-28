# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.apps.llamaindex.llama` instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.apps.llamaindex.llama import COMPONENT_VIEWS
from trulens.apps.llamaindex.llama import LLM
from trulens.apps.llamaindex.llama import Agent
from trulens.apps.llamaindex.llama import Other
from trulens.apps.llamaindex.llama import Prompt
from trulens.apps.llamaindex.llama import Tool
from trulens.apps.llamaindex.llama import component_of_json
from trulens.apps.llamaindex.llama import constructor_of_class

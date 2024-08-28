# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.instrument.langchain.langchain` instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.instrument.langchain.langchain import COMPONENT_VIEWS
from trulens.instrument.langchain.langchain import LLM
from trulens.instrument.langchain.langchain import Other
from trulens.instrument.langchain.langchain import Prompt
from trulens.instrument.langchain.langchain import component_of_json
from trulens.instrument.langchain.langchain import constructor_of_class

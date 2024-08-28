# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.apps.langchain.langchain` instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.apps.langchain.langchain import COMPONENT_VIEWS
from trulens.apps.langchain.langchain import LLM
from trulens.apps.langchain.langchain import Other
from trulens.apps.langchain.langchain import Prompt
from trulens.apps.langchain.langchain import component_of_json
from trulens.apps.langchain.langchain import constructor_of_class

# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.app.base`,
    `trulens.instrument.langchain.langchain`, or
    `trulens.instrument.llamaindex.llama` instead.

"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.app import ATTRIBUTE_ERROR_MESSAGE
from trulens.core.app import CLASS_INFO
from trulens.core.app import JSON_BASES
from trulens.core.app import LLM
from trulens.core.app import Agent
from trulens.core.app import App
from trulens.core.app import ComponentView
from trulens.core.app import CustomComponent
from trulens.core.app import Memory
from trulens.core.app import Other
from trulens.core.app import Prompt
from trulens.core.app import RecordingContext
from trulens.core.app import Tool
from trulens.core.app import TrulensComponent
from trulens.core.app import instrumented_component_views
from trulens.instrument.langchain.langchain import LangChainComponent
from trulens.instrument.llamaindex.llama import LlamaIndexComponent
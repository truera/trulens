# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.feedback.provider` or `trulens.feedback.llm_provider` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.feedback.provider import Provider
from trulens.feedback.llm_provider import LLMProvider

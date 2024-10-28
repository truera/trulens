# ruff: noqa: E402, F403, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.feedback.feedback` or `trulens.core.feedback.feedback` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.feedback import InvalidSelector
from trulens.core.feedback.feedback import SkipEval
from trulens.feedback.feedback import rag_triad

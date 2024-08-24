# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.dashboard.display` or `trulens.core.utils.trulens` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils.trulens import get_feedback_result
from trulens.dashboard.display import get_icon

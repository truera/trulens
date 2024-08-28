# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.core.session` or `trulens.dashboard.run` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.session import TruSession as Tru
from trulens.core.utils.text import format_seconds as humanize_seconds
from trulens.dashboard.run import DASHBOARD_START_TIMEOUT

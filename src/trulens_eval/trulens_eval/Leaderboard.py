# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.dashboard.pages.Leaderboard` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.dashboard.Leaderboard import MIGRATION_UNKNOWN_STR
from trulens.dashboard.Leaderboard import leaderboard

# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.dashboard.Leaderboard` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.database.legacy.migration import MIGRATION_UNKNOWN_STR
from trulens.dashboard.Leaderboard import render_leaderboard as leaderboard

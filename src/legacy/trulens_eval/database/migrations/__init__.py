# ruff: noqa: E402, F403, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.database.migrations` instead.
"""

from trulens.core.utils.deprecation import packages_dep_warn

# NOTE: This file had contents in trulens_eval none were public or aliases.
# Because of that, this backwards compatibility module is empty.

packages_dep_warn("trulens_eval.database.migrations")

from trulens.core.database.migrations import *

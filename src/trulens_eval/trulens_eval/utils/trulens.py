# ruff: noqa: E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.utils.trulens`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils.trulens import *

# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.dashboard.appui`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.dashboard.appui import VALUE_MAX_CHARS
from trulens.dashboard.appui import AppUI
from trulens.dashboard.appui import RecordWidget
from trulens.dashboard.appui import Selector
from trulens.dashboard.appui import SelectorValue
from trulens.dashboard.appui import debug_style

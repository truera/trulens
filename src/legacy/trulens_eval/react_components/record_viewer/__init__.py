# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.dashboard.react_components.record_viewer` instead.
"""

from trulens.core.utils import deprecation
from trulens.core.utils.deprecation import packages_dep_warn

packages_dep_warn("trulens_eval.react_components.record_viewer")

from trulens.dashboard.react_components.record_viewer import record_viewer

record_viewer = deprecation.function_moved(
    record_viewer,
    old="trulens_eval.react_components.record_viewer",
    new="trulens.dashboard.react_components.record_viewer",
)

# ruff: noqa: E402
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.dashboard.react_components.record_viewer` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn(
    "trulens_eval.react_components.record_viewer"
)

from trulens.dashboard.components.record_viewer import record_viewer

record_viewer = deprecation_utils.function_moved(
    record_viewer,
    old="trulens_eval.react_components.record_viewer",
    new="trulens.dashboard.components.record_viewer",
)

__all__ = ["record_viewer"]

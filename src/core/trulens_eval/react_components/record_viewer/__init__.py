"""Aliases.

Aliases to new locations of various names. This module is deprecated and will be
removed in a future release.
"""

from trulens_eval import dep_warn
from trulens.core.utils import deprecation

dep_warn("trulens_eval.react_components.record_viewer")

from trulens.dashboard.react_components.record_viewer import record_viewer

record_viewer = deprecation.function_moved(
    record_viewer,
    old="trulens_eval.react_components.record_viewer",
    new="trulens.dashboard.react_components.record_viewer")

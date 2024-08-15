# ruff: noqa: E402, F822

from typing import TYPE_CHECKING

from trulens.core.utils import imports as import_utils

if TYPE_CHECKING:
    # Needed for static tools to resolve submodules:
    from trulens.dashboard.run import run_dashboard
    from trulens.dashboard.run import stop_dashboard

_FUNCTIONS = {
    "run_dashboard": ("trulens-dashboard", "trulens.dashboard.run"),
    "stop_dashboard": ("trulens-dashboard", "trulens.dashboard.run"),
}

_KINDS = {
    "function": _FUNCTIONS,
}

help, help_str = import_utils.make_help_str(_KINDS)

__getattr__ = import_utils.make_getattr_override(_KINDS, help_str=help_str)

__all__ = [
    "run_dashboard",
    "stop_dashboard",
]

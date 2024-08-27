# ruff: noqa: E402, F822

from typing import TYPE_CHECKING

from trulens.auto._utils import auto as auto_utils

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

__getattr__ = auto_utils.make_getattr_override(
    doc="TruLens dashboard functions.", kinds=_KINDS
)

__all__ = [
    "run_dashboard",
    "stop_dashboard",
]

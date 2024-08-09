from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.dashboard.run import run_dashboard
from trulens.dashboard.run import stop_dashboard

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "run_dashboard",
    "stop_dashboard",
]

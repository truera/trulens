from importlib.metadata import version

from trulens.dashboard.run import run_dashboard
from trulens.dashboard.run import stop_dashboard

__version__ = version(__package__ or __name__)

__all__ = [
    "run_dashboard",
    "stop_dashboard",
]

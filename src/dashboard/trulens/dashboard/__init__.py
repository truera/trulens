# WARNING: This file does not follow the no-init aliases import standard.


from importlib.metadata import version

from trulens.core.utils import imports as import_utils
from trulens.dashboard.run import run_dashboard
from trulens.dashboard.run import run_dashboard_sis
from trulens.dashboard.run import stop_dashboard

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)

__all__ = ["run_dashboard", "stop_dashboard", "run_dashboard_sis"]

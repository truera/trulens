"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-autogen` package installed.

    ```bash
    pip install trulens-apps-autogen
    ```
"""

from importlib.metadata import version

from trulens.apps.autogen.tru_autogen import AutoGenInstrument
from trulens.apps.autogen.tru_autogen import TruAutoGen
from trulens.core.utils import imports as import_utils

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)


__all__ = ["TruAutoGen", "AutoGenInstrument"]

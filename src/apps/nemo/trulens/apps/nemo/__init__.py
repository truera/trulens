"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-nemo` package installed.

    ```bash
    pip install trulens-apps-nemo
    ```
"""

from importlib.metadata import version

from trulens.apps.nemo.tru_rails import RailsActionSelect
from trulens.apps.nemo.tru_rails import RailsInstrument
from trulens.apps.nemo.tru_rails import TruRails
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = ["TruRails", "RailsInstrument", "RailsActionSelect"]

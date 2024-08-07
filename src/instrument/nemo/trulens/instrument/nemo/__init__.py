"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-nemo` package installed.

    ```bash
    pip install trulens-instrument-nemo
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.instrument.nemo.tru_rails import RailsActionSelect
from trulens.instrument.nemo.tru_rails import RailsInstrument
from trulens.instrument.nemo.tru_rails import TruRails

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = ["TruRails", "RailsInstrument", "RailsActionSelect"]

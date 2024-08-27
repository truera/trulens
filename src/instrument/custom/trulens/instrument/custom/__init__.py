"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-custom` package installed.

    ```bash
    pip install trulens-instrument-custom
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.instrument.custom.custom import TruCustomApp
from trulens.instrument.custom.custom import instrument

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "TruCustomApp",
    "instrument",
]

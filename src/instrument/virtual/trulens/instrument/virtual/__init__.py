"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-virtual` package installed.

    ```bash
    pip install trulens-instrument-virtual
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.instrument.virtual.virtual import TruVirtual
from trulens.instrument.virtual.virtual import VirtualApp
from trulens.instrument.virtual.virtual import VirtualRecord

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "TruVirtual",
    "VirtualApp",
    "VirtualRecord",
]

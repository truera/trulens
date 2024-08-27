"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-basic` package installed.

    ```bash
    pip install trulens-instrument-basic
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.instrument.basic.basic import TruBasicApp
from trulens.instrument.basic.basic import TruBasicCallableInstrument
from trulens.instrument.basic.basic import TruWrapperApp

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "TruBasicApp",
    "TruBasicCallableInstrument",
    "TruWrapperApp",
]

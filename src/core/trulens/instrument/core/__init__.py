"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-custom` package installed.

    ```bash
    pip install trulens-instrument-custom
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.instrument.core.basic import TruBasicApp
from trulens.instrument.core.basic import TruBasicCallableInstrument
from trulens.instrument.core.basic import TruWrapperApp
from trulens.instrument.core.custom import TruCustomApp
from trulens.instrument.core.custom import instrument
from trulens.instrument.core.virtual import TruVirtual
from trulens.instrument.core.virtual import VirtualApp
from trulens.instrument.core.virtual import VirtualRecord

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "TruCustomApp",
    "instrument",
    "TruBasicApp",
    "TruBasicCallableInstrument",
    "TruWrapperApp",
    "TruVirtual",
    "VirtualApp",
    "VirtualRecord",
]

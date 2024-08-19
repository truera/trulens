"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-cortex` package installed.

    ```bash
    pip install trulens-providers-cortex
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.providers.cortex.provider import Cortex

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "Cortex",
]

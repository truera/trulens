"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-cortex` package installed.

    ```bash
    pip install trulens-providers-cortex
    ```
"""

from importlib.metadata import version

from trulens.providers.cortex.provider import Cortex

__version__ = version(__package__ or __name__)

__all__ = [
    "Cortex",
]

"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-benchmark` package installed.

    ```bash
    pip install trulens-benchmark
    ```
"""

from importlib.metadata import version

__version__ = version(__package__ or __name__)

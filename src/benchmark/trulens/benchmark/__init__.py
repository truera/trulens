"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-benchmark` package installed.

    ```bash
    pip install trulens-benchmark
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))

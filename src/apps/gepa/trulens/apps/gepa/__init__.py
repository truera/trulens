"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-gepa` package installed.

    ```bash
    pip install trulens-apps-gepa
    ```
"""

from importlib.metadata import version

from trulens.apps.gepa.fitness import TruLensFitness
from trulens.apps.gepa.fitness import run_evolution
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))

__all__ = [
    "TruLensFitness",
    "run_evolution",
]

"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-dspy` package installed.

    ```bash
    pip install trulens-apps-dspy
    ```

    The adapter does not depend on DSPy itself; install `dspy` alongside it to
    run DSPy evaluations and optimizers.
"""

from importlib.metadata import version

from trulens.apps.dspy.metric import TruDSPyMetric
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))

__all__ = [
    "TruDSPyMetric",
]

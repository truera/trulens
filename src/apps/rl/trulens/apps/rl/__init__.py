"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-rl` package installed.

    ```bash
    pip install trulens-apps-rl
    ```
"""

from importlib.metadata import version

from trulens.apps.rl.reward import RewardFunction
from trulens.apps.rl.reward import TRLRewardAdapter
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))

__all__ = [
    "RewardFunction",
    "TRLRewardAdapter",
]

"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-connectors-snowflake` package installed.

    ```bash
    pip install trulens-connectors-snowflake
    ```
"""

# WARNING: This file does not follow the no-init aliases import standard.

from importlib.metadata import version

from trulens.connectors.snowflake.connector import SnowflakeConnector
from trulens.core.utils import imports as import_utils

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)

__all__ = [
    "SnowflakeConnector",
]

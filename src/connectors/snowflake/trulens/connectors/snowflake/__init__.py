"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-connectors-snowflake` package installed.

    ```bash
    pip install trulens-connectors-snowflake
    ```
"""

from importlib.metadata import version

from trulens.connectors.snowflake.connector import SnowflakeConnector
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "SnowflakeConnector",
]

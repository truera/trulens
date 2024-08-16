"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-workspaces-snowflake` package installed.

    ```bash
    pip install trulens-workspaces-snowflake
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.workspaces.snowflake.workspace import SnowflakeWorkspace

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "SnowflakeWorkspace",
]

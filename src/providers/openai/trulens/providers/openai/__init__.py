"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-openai` package installed.

    ```bash
    pip install trulens-providers-openai
    ```
"""

from importlib.metadata import version

from trulens.core.utils import imports as import_utils

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)

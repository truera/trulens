"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-openai` package installed.

    ```bash
    pip install trulens-providers-openai
    ```
"""
# WARNING: This file does not follow the no-init aliases import standard.

from importlib.metadata import version

from trulens.core.utils import imports as import_utils
from trulens.providers.openai.provider import AzureOpenAI
from trulens.providers.openai.provider import OpenAI

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)

__all__ = ["OpenAI", "AzureOpenAI"]

"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-openai` package installed.

    ```bash
    pip install trulens-providers-openai
    ```
"""

from importlib.metadata import version

from trulens.providers.openai.provider import AzureOpenAI
from trulens.providers.openai.provider import OpenAI

__version__ = version(__package__ or __name__)

__all__ = ["OpenAI", "AzureOpenAI"]

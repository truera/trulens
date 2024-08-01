"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-litellm` package installed.

    ```bash
    pip install trulens-providers-litellm
    ```
"""

from importlib.metadata import version

from trulens.providers.litellm.provider import LiteLLM

__version__ = version(__package__ or __name__)

__all__ = [
    "LiteLLM",
]

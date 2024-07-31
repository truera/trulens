"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-litellm` package installed.

    ```bash
    pip install trulens-providers-litellm
    ```
"""

from trulens.providers.litellm.provider import LiteLLM

__all__ = [
    "LiteLLM",
]

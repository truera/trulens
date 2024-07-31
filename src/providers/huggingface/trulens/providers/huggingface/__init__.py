"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-huggingface` package installed.

    ```bash
    pip install trulens-providers-huggingface
    ```
"""

from trulens.providers.huggingface.provider import Huggingface

__all__ = [
    "Huggingface",
]

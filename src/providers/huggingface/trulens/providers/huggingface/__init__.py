"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-huggingface` package installed.

    ```bash
    pip install trulens-providers-huggingface
    ```
"""

from importlib.metadata import version

from trulens.providers.huggingface.provider import Huggingface
from trulens.providers.huggingface.provider import HuggingfaceLocal

__version__ = version(__package__ or __name__)

__all__ = ["Huggingface", "HuggingfaceLocal"]

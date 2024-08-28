"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-llamaindex` package installed.

    ```bash
    pip install trulens-apps-llamaindex
    ```
"""

from importlib.metadata import version

from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes
from trulens.apps.llamaindex.llama import LlamaIndexComponent
from trulens.apps.llamaindex.tru_llama import LlamaInstrument
from trulens.apps.llamaindex.tru_llama import TruLlama
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "TruLlama",
    "LlamaInstrument",
    "WithFeedbackFilterNodes",
    "LlamaIndexComponent",
]

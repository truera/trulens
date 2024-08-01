"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-llamaindex` package installed.

    ```bash
    pip install trulens-instrument-llamaindex
    ```
"""

from importlib.metadata import version

from trulens.instrument.llamaindex.guardrails import WithFeedbackFilterNodes
from trulens.instrument.llamaindex.llama import LlamaIndexComponent
from trulens.instrument.llamaindex.tru_llama import LlamaInstrument
from trulens.instrument.llamaindex.tru_llama import TruLlama

__version__ = version(__package__ or __name__)

__all__ = [
    "TruLlama",
    "LlamaInstrument",
    "WithFeedbackFilterNodes",
    "LlamaIndexComponent",
]

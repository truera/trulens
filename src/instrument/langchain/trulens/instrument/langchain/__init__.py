"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-langchain` package installed.

    ```bash
    pip install trulens-instrument-langchain
    ```
"""

from trulens.instrument.langchain.guardrails import WithFeedbackFilterDocuments
from trulens.instrument.langchain.langchain import LangChainComponent
from trulens.instrument.langchain.tru_chain import LangChainInstrument
from trulens.instrument.langchain.tru_chain import TruChain

__all__ = [
    "TruChain",
    "LangChainInstrument",
    "WithFeedbackFilterDocuments",
    "LangChainComponent",
]

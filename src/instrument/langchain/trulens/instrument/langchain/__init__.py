"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-langchain` package installed.

    ```bash
    pip install trulens-instrument-langchain
    ```
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.instrument.langchain.guardrails import WithFeedbackFilterDocuments
from trulens.instrument.langchain.langchain import LangChainComponent
from trulens.instrument.langchain.tru_chain import LangChainInstrument
from trulens.instrument.langchain.tru_chain import TruChain

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "TruChain",
    "LangChainInstrument",
    "WithFeedbackFilterDocuments",
    "LangChainComponent",
]

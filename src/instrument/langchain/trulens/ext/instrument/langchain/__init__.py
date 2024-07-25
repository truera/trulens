import importlib.metadata

from trulens.ext.instrument.langchain.guardrails import (
    WithFeedbackFilterDocuments,
)
from trulens.ext.instrument.langchain.tru_chain import LangChainInstrument
from trulens.ext.instrument.langchain.tru_chain import TruChain

__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    "TruChain",
    "LangChainInstrument",
    "WithFeedbackFilterDocuments",
]

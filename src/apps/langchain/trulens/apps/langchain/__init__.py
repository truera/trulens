"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-langchain` package installed.

    ```bash
    pip install trulens-apps-langchain
    ```
"""

from importlib.metadata import version

from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments
from trulens.apps.langchain.langchain import LangChainComponent
from trulens.apps.langchain.tru_chain import LangChainInstrument
from trulens.apps.langchain.tru_chain import TruChain
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "TruChain",
    "LangChainInstrument",
    "WithFeedbackFilterDocuments",
    "LangChainComponent",
]

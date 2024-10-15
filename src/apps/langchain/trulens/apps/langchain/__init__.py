"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-langchain` package installed.

    ```bash
    pip install trulens-apps-langchain
    ```
"""

# WARNING: This file does not follow the no-init aliases import standard.

from importlib.metadata import version

from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments
from trulens.apps.langchain.langchain import LangChainComponent
from trulens.apps.langchain.tru_chain import LangChainInstrument
from trulens.apps.langchain.tru_chain import TruChain
from trulens.core.utils import imports as import_utils

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)


__all__ = [
    "TruChain",
    "LangChainInstrument",
    "WithFeedbackFilterDocuments",
    "LangChainComponent",
]

"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-apps-langgraph` package installed.

    ```bash
    pip install trulens-apps-langgraph
    ```
"""

# WARNING: This file does not follow the no-init aliases import standard.

from importlib.metadata import version

from trulens.apps.langgraph.tru_graph import LangGraphInstrument
from trulens.apps.langgraph.tru_graph import TruGraph
from trulens.apps.langgraph.tru_graph import instrument_task
from trulens.core.utils import imports as import_utils

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)


__all__ = [
    "TruGraph",
    "LangGraphInstrument",
    "instrument_task",
]

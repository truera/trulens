"""TruLens LangGraph instrumentation."""

from importlib.metadata import version

from trulens.apps.langgraph.tru_graph import LangGraphInstrument
from trulens.apps.langgraph.tru_graph import TruGraph
from trulens.core.utils import imports as import_utils

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)


__all__ = ["TruGraph", "LangGraphInstrument"]

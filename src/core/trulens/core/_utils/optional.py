"""Private optional import utils.

Not for use outside trulens namespace.
"""

from trulens.core.utils import imports as import_utils

# Optional app types:
REQUIREMENT_APPS_LLAMA = import_utils.format_import_errors(
    "trulens-apps-llamaindex", purpose="instrumenting LlamaIndex apps"
)
REQUIREMENT_APPS_LANGCHAIN = import_utils.format_import_errors(
    "trulens-apps-langchain", purpose="instrumenting LangChain apps"
)
REQUIREMENT_APPS_LANGGRAPH = import_utils.format_import_errors(
    "trulens-apps-langgraph", purpose="instrumenting LangGraph apps"
)
REQUIREMENT_APPS_NEMO = import_utils.format_import_errors(
    "trulens-apps-nemo", purpose="instrumenting NeMo Guardrails apps"
)

REQUIREMENT_TQDM = import_utils.format_import_errors(
    "tqdm", purpose="displaying progress bars"
)

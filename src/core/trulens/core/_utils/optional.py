"""Private optional import utils.

Not for use outside trulens namespace.
"""

from trulens.core.utils.imports import format_import_errors

# Optional app types:
REQUIREMENT_APPS_LLAMA = format_import_errors(
    "trulens-apps-llamaindex", purpose="instrumenting LlamaIndex apps"
)
REQUIREMENT_APPS_LANGCHAIN = format_import_errors(
    "trulens-apps-langchain", purpose="instrumenting LangChain apps"
)
REQUIREMENT_APPS_NEMO = format_import_errors(
    "trulens-apps-nemo", purpose="instrumenting NeMo Guardrails apps"
)

REQUIREMENT_TQDM = format_import_errors(
    "tqdm", purpose="displaying progress bars"
)

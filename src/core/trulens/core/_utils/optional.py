"""Private optional import utils.

Not for use outside trulens namespace.
"""

from trulens.core.utils.imports import format_import_errors

# Optional app types:
REQUIREMENT_INSTRUMENT_LLAMA = format_import_errors(
    "trulens-apps-llamaindex", purpose="instrumenting LlamaIndex apps"
)
REQUIREMENT_INSTRUMENT_LANGCHAIN = format_import_errors(
    "trulens-apps-langchain", purpose="instrumenting LangChain apps"
)
REQUIREMENT_INSTRUMENT_NEMO = format_import_errors(
    "trulens-apps-nemo", purpose="instrumenting NeMo Guardrails apps"
)

REQUIREMENT_SNOWFLAKE = format_import_errors(
    [
        "snowflake-core",
        "snowflake-connector-python",
        "snowflake-snowpark-python",
        "snowflake-sqlalchemy",
    ],
    purpose="connecting to Snowflake",
)

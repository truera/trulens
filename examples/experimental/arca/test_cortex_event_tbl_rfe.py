# Lightweight script to fetch events (trace spans) from the Snowflake AI Observability Event Table and run reference-free GPA evals.
import os

from dotenv import load_dotenv
from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession
from trulens.core.feedback.selector import Trace
from trulens.providers.cortex.provider import Cortex

load_dotenv()

APP_NAME = ...  # REPLACE with Cortex Agent Name (e.g. "PROMPT_OPTIM_AGENT")
RECORD_ID = ...  # REPLACE with Cortex Agent Request ID (e.g. c728452a-ab78-4657-a5a2-37539a39aef2)

# Authenticate with Snowflake
connection_parameters = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PAT"),
    # MUST be set to DB that the Agent is deployed to
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    # MUST be set to SCHEMA that the Agent is deployed to
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

snowpark_session = Session.builder.configs(connection_parameters).create()
tru_session = TruSession(SnowflakeConnector(snowpark_session=snowpark_session))
cortex_provider = Cortex(snowpark_session)


# Get agent events from Snowflake AI Observability Event Table
events_df = tru_session.get_events(
    app_name=APP_NAME, app_version=None, record_ids=[RECORD_ID]
)
trace = Trace()
trace.events = events_df

# Run reference-free GPA evals
# The return format is a tuple of float and dict, the dict contains a single key "reason" with a string value containing the reasoning behind the score.

score, reason_dict = cortex_provider.logical_consistency_with_cot_reasons(
    trace=trace
)

print(f"Logical Consistency Score: {score}")
print(f"Logical Consistency Reason: {reason_dict['reason']}")

# cortex_provider.execution_efficiency_with_cot_reasons(trace=trace)
# cortex_provider.plan_adherence_with_cot_reasons(trace=trace)
# cortex_provider.plan_quality_with_cot_reasons(trace=trace)


# NOTE:
# You can also run the GPA evals on string-based traces, however you will not get the benefit of trace compression, and the GPA evals will (1) be more expensive and (2) take longer to run.
# GPA evals on string-based traces are still supported for backwards compatibility, but we recommend using the Trace object instead.

# NOTE:
# GPA evals have been tested and evaluated for single-turn conversations, not multi-turn conversations (yet).
# You may still use traces from multi-turn conversations, but you will need to manually feed each trace into the GPA evals.
# If you wanted to experiment with feeding multiple traces at one time (from a multi-turn conversation), you could compress the n>1 traces from your multi-turn conversation into a single trace (this has not been tested).

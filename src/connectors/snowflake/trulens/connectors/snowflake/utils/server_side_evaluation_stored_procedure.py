from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import session as core_session
from trulens.core.schema import feedback as feedback_schema
from trulens.providers.cortex import provider as cortex_provider

from snowflake.snowpark import Session


def run(snowpark_session: Session):
    # Ensure any Cortex provider uses the only Snowflake connection allowed in this stored procedure.
    cortex_provider._SNOWFLAKE_STORED_PROCEDURE_CONNECTION = (
        snowpark_session.connection
    )
    # Run deferred feedback evaluator.
    connector = SnowflakeConnector(
        snowpark_session=snowpark_session,
        init_server_side=False,
        database_check_revision=False,  # TODO: check revision in the future?
    )
    tru_session = core_session.TruSession(connector)
    tru_session.start_evaluator(
        run_location=feedback_schema.FeedbackRunLocation.SNOWFLAKE,
        return_when_done=True,
        disable_tqdm=True,
    )

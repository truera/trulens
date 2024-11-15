from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import session as core_session
from trulens.core.schema import feedback as feedback_schema
from trulens.providers.cortex import provider as cortex_provider

from snowflake.snowpark import Session
from snowflake.snowpark import context


def run(snowpark_session: Session):
    # context.get_active_session() will fail if there is no or more than one active session. This is not a concern
    # for server side eval in the warehouse as there should only be only active session in the execution context.
    cortex_provider._SNOWFLAKE_STORED_PROCEDURE_SESSION = (
        context.get_active_session()
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

from trulens.core import TruSession
from trulens.core.schema.feedback import FeedbackRunLocation
import trulens.providers.cortex.provider

from snowflake.snowpark import Session
from snowflake.sqlalchemy import URL


def run(snowpark_session: Session):
    # Set up sqlalchemy engine parameters.
    conn = snowpark_session.connection
    engine_params = {}
    engine_params["paramstyle"] = "qmark"
    engine_params["creator"] = lambda: conn
    database_args = {"engine_params": engine_params}
    # Ensure any Cortex provider uses the only Snowflake connection allowed in this stored procedure.
    trulens.providers.cortex.provider._SNOWFLAKE_STORED_PROCEDURE_CONNECTION = (
        conn
    )
    # Run deferred feedback evaluator.
    db_url = URL(
        account=snowpark_session.get_current_account(),
        user=snowpark_session.get_current_user(),
        password="password",
        database=snowpark_session.get_current_database(),
        schema=snowpark_session.get_current_schema(),
        warehouse=snowpark_session.get_current_warehouse(),
        role=snowpark_session.get_current_role(),
    )
    tru_session = TruSession(
        database_url=db_url,
        database_check_revision=False,  # TODO: check revision in the future?
        database_args=database_args,
    )
    tru_session.start_evaluator(
        run_location=FeedbackRunLocation.SNOWFLAKE,
        return_when_done=True,
        disable_tqdm=True,
    )

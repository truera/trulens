from snowflake.snowpark.context import get_active_session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession
from trulens.core.experimental import Feature
from trulens.dashboard.Leaderboard import leaderboard_main

if __name__ == "__main__":
    session = get_active_session()
    # st.write(session)

    tru_session = TruSession(
        connector=SnowflakeConnector(
            snowpark_session=session, database_check_revision=False
        )
    )
    tru_session.experimental_enable_feature(Feature.SIS_COMPATIBILITY)
    leaderboard_main()

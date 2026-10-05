from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession
from trulens.apps.app import TruApp

from src.agent.app import AgentApp
from src.eval.metrics import build_metrics
from src.services.config import get_snowpark_session


def setup_observability():
    """Initialize TruLens observability with Snowflake connector and all metrics."""
    snowpark_session = get_snowpark_session()
    sf_connector = SnowflakeConnector(snowpark_session=snowpark_session)
    session = TruSession(connector=sf_connector)

    agent_app = AgentApp()
    all_metrics = build_metrics(snowpark_session)

    tru_app = TruApp(
        agent_app,
        app_name="Support Cloud Agent",
        app_version="v2",
        connector=sf_connector,
        main_method=agent_app.ask,
    )

    return agent_app, tru_app, session, sf_connector, all_metrics

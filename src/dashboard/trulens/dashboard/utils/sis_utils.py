"""This module contains utility functions for rendering the dashboard on Streamlit in Snowflake."""

from trulens.core import TruSession
from trulens.core.experimental import Feature
from trulens.core.utils import imports as import_utils
from trulens.dashboard.Leaderboard import leaderboard_main
from trulens.dashboard.pages.Compare import compare_main
from trulens.dashboard.pages.Records import records_main

with import_utils.OptionalImports(
    messages=import_utils.format_import_errors(
        "snowflake-connector-python",
        purpose="creating or retrieving the Snowflake Session",
    )
):
    from snowflake.snowpark.context import get_active_session

with import_utils.OptionalImports(
    messages=import_utils.format_import_errors(
        "trulens-connectors-snowflake",
        purpose="setting up the dashboard on Streamlit in Snowflake",
    )
):
    from trulens.connectors.snowflake import SnowflakeConnector

DASHBOARD_PAGES = {
    "leaderboard": leaderboard_main,
    "records": records_main,
    "compare": compare_main,
}


def _setup_tru_session():
    session = get_active_session()

    tru_session = TruSession(
        connector=SnowflakeConnector(
            snowpark_session=session, database_check_revision=False
        )
    )
    tru_session.experimental_enable_feature(Feature.SIS_COMPATIBILITY)


def render_page(page_name: str):
    _setup_tru_session()

    if page_name in DASHBOARD_PAGES:
        DASHBOARD_PAGES[page_name]()
    else:
        raise ValueError(f"Unknown page name: {page_name}")

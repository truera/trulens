"""
Main entry point for the TruLens dashboard using st.navigation and st.Page.
"""

import os
from pathlib import Path
import threading
import time
import uuid

import streamlit as st
from trulens.core.session import TruSession
from trulens.dashboard.utils.dashboard_utils import get_session
from trulens.dashboard.utils.dashboard_utils import set_page_config


def ping_session(tru_session: TruSession):
    snowpark_session = tru_session.connector.snowpark_session
    original_snowpark_connection = snowpark_session.connection
    original_snowpark_connection_cursor = original_snowpark_connection.cursor()

    def ping():
        while True:
            try:
                print("--------------------------------")
                print("SELECT XYZ:")
                print(snowpark_session.sql("SELECT 1").collect()[0])
                print(
                    snowpark_session.connection.cursor()
                    .execute("SELECT 2")
                    .fetchone()
                )
                print(
                    original_snowpark_connection.cursor()
                    .execute("SELECT 3")
                    .fetchone()
                )
                print(
                    original_snowpark_connection_cursor.execute(
                        "SELECT 4"
                    ).fetchone()
                )
                print(
                    original_snowpark_connection_cursor.execute(
                        f"SELECT '{str(uuid.uuid4())}'"
                    ).fetchone()
                )
                print(
                    tru_session.get_events(
                        app_name=str(uuid.uuid4()),
                        app_version=str(uuid.uuid4()),
                    ),
                )
                print("Num apps:", len(tru_session.get_apps()))
                import sqlalchemy as sa

                with tru_session.connector.db.session.begin() as session:
                    q = sa.select(tru_session.connector.db.orm.AppDefinition)
                    q = q.filter_by(app_name=str(uuid.uuid4()))
                    print([list(row[0]) for row in session.execute(q)])
                print("--------------------------------XYZ")
            except Exception as e:
                print(f"Ping error: {e}")
            time.sleep(60)

    # Start ping in a separate daemon thread
    ping_thread = threading.Thread(target=ping, daemon=True)
    ping_thread.start()


def main():
    """Main dashboard function using st.navigation and st.Page."""
    tru_session = get_session()
    ping_session(tru_session)
    print("KOJIKUN XYZ 1")
    print("KOJIKUN XYZ 2")
    print("KOJIKUN XYZ 3")
    set_page_config(page_title="Dashboard")
    tabs_dir = Path(__file__).parent / "tabs"
    pages = [
        st.Page(str(tabs_dir / "Leaderboard.py"), default=True),
        st.Page(str(tabs_dir / "Records.py")),
        st.Page(str(tabs_dir / "Compare.py")),
    ]
    if custom_pages_dir := os.environ.get("TRULENS_UI_CUSTOM_PAGES"):
        if os.path.isdir(custom_pages_dir):
            for file in os.listdir(custom_pages_dir):
                if file.endswith(".py"):
                    pages.append(st.Page(os.path.join(custom_pages_dir, file)))
        else:
            st.error(
                f"TRULENS_UI_CUSTOM_PAGES is set to {custom_pages_dir} but it is not a directory!"
            )
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()

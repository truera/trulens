"""
Main entry point for the TruLens dashboard using st.navigation and st.Page.
"""

import streamlit as st
from trulens.dashboard.tabs.Compare import compare_page as _compare_page
from trulens.dashboard.tabs.Leaderboard import (
    leaderboard_page as _leaderboard_page,
)
from trulens.dashboard.tabs.Records import records_page as _records_page
from trulens.dashboard.utils.dashboard_utils import get_session
from trulens.dashboard.utils.dashboard_utils import set_page_config


def leaderboard_page():
    """Leaderboard page function for st.Page."""
    _leaderboard_page()


def records_page():
    """Records page function for st.Page."""
    _records_page()


def compare_page():
    """Compare page function for st.Page."""
    _compare_page()


def stability_page():
    """Stability page function for st.Page."""
    st.info("🚧 Stability page - placeholder for now")


def main():
    """Main dashboard function using st.navigation and st.Page."""
    get_session()
    set_page_config(page_title="Dashboard")
    pages = [
        st.Page(leaderboard_page, title="Leaderboard", icon="🏆", default=True),
        st.Page(records_page, title="Records", icon="📝"),
        st.Page(compare_page, title="Compare", icon="⚖️"),
    ]
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()

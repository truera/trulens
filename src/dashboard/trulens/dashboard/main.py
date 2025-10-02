"""
Main entry point for the TruLens dashboard using st.navigation and st.Page.
"""

import os
from pathlib import Path

import streamlit as st
from trulens.dashboard.utils.dashboard_utils import get_session
from trulens.dashboard.utils.dashboard_utils import set_page_config


def main():
    """Main dashboard function using st.navigation and st.Page."""
    get_session()
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

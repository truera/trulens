"""Navigation system for the TruLens dashboard with custom page support."""

from typing import Callable, Dict

import streamlit as st
from trulens.dashboard.Leaderboard import render_leaderboard
from trulens.dashboard.pages.Compare import render_app_comparison
from trulens.dashboard.pages.Records import render_records
from trulens.dashboard.registry import get_registered_pages
from trulens.dashboard.registry import load_custom_pages
from trulens.dashboard.utils import dashboard_utils


def create_navigation() -> Dict[str, Callable]:
    """Create navigation with built-in and custom pages.

    Returns:
        Dictionary mapping page names to their render functions.
    """
    # Load custom pages from configuration.
    load_custom_pages()
    # Built-in pages.
    pages = {
        "Leaderboard": render_leaderboard,
        "Compare": render_app_comparison,
        "Records": render_records,
    }
    # Load built-in plugins (like Stability) - import triggers auto-registration
    from trulens.dashboard import plugins  # noqa: F401

    # Add custom pages.
    custom_pages = get_registered_pages()
    pages.update(custom_pages)
    return pages


def render_navigation() -> None:
    """Render the main navigation and page content."""
    dashboard_utils.set_page_config(page_title="TruLens Dashboard")
    # Get all pages.
    pages = create_navigation()
    page_names = list(pages.keys())
    # Use query params for page selection.
    selected_page = st.query_params.get("page", page_names[0])
    if selected_page not in page_names:
        selected_page = page_names[0]
    # Create tabs.
    tabs = st.tabs(page_names)
    # Render the selected page content in its tab.
    for i, (page_name, page_func) in enumerate(pages.items()):
        with tabs[i]:
            if (
                page_name == selected_page
                or st.session_state.get("current_tab") == i
            ):
                # Get app name from sidebar (common to all pages).
                app_name = dashboard_utils.render_sidebar()
                if app_name:
                    page_func(app_name)
                elif page_name == "Leaderboard":
                    # Leaderboard can work without app selection
                    page_func(None)

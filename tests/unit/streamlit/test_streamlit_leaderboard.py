"""Test basic UI components for the TruLens Dashboard Leaderboard page using AppTest."""

from unittest.mock import patch

import pytest
import trulens.dashboard.constants as dashboard_constants

from tests.unit.streamlit.test_streamlit_utils import AppTestHelper
from tests.unit.streamlit.test_streamlit_utils import MockManager
from tests.unit.streamlit.test_streamlit_utils import create_mock_data_dict


class TestLeaderboardUI:
    """Test suite for Leaderboard UI components using AppTest."""

    @pytest.fixture(scope="class")
    def mock_data(self):
        """Mock data for testing - class scoped for reuse."""
        return create_mock_data_dict()

    def test_init_page_state(self):
        """Test that page state initialization works correctly."""
        with patch(
            "trulens.dashboard.utils.dashboard_utils.read_query_params_into_session_state"
        ):

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import init_page_state

                init_page_state()

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_render_leaderboard_no_app_versions(self):
        """Test leaderboard rendering when no app versions are found."""
        with MockManager.mock_dashboard_utils(empty_versions=True):

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import (
                    render_leaderboard,
                )

                render_leaderboard("Test App")

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_has_error_with_message(
                app, "No app versions found"
            )

    def test_render_leaderboard_no_records(self):
        """Test leaderboard rendering when no records are found."""
        with MockManager.mock_dashboard_utils(empty_records=True):

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import (
                    render_leaderboard,
                )

                render_leaderboard("Test App")

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_render_leaderboard_with_data(self, mock_data):
        """Test leaderboard rendering with valid data."""
        with MockManager.mock_all_common_dependencies(mock_data):

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import (
                    render_leaderboard,
                )

                render_leaderboard("Test App")

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)
            AppTestHelper.assert_has_title_content(
                app, dashboard_constants.LEADERBOARD_PAGE_NAME
            )
            assert len(app.markdown) > 0

    def test_render_plot_tab_no_feedback(self):
        """Test plot tab rendering when no feedback functions exist."""

        def test_app():
            import pandas as pd
            from trulens.dashboard.tabs.Leaderboard import _render_plot_tab

            df = pd.DataFrame({"app_id": ["app_1"], "app_version": ["v1.0"]})
            feedback_col_names = []
            _render_plot_tab(df, feedback_col_names)

        app = AppTestHelper.create_and_run_app(test_app)
        AppTestHelper.assert_has_warning_with_message(
            app, "No feedback functions found"
        )

    def test_render_plot_tab_with_feedback(self):
        """Test plot tab rendering with feedback data."""

        def test_app():
            from trulens.dashboard.tabs.Leaderboard import _render_plot_tab

            from tests.unit.streamlit.test_streamlit_utils import (
                TestDataFactory,
            )

            df = TestDataFactory.create_processed_df(include_feedback=True)
            feedback_col_names = ["feedback_score", "relevance"]
            _render_plot_tab(df, feedback_col_names)

        app = AppTestHelper.create_and_run_app(test_app)
        AppTestHelper.assert_no_errors(app)

    def test_render_list_tab(self):
        """Test list tab rendering with app data."""

        def test_app():
            from trulens.dashboard.tabs.Leaderboard import _render_list_tab

            from tests.unit.streamlit.test_streamlit_utils import (
                TestDataFactory,
            )

            df = TestDataFactory.create_processed_df()
            feedback_col_names = ["feedback_score"]
            feedback_directions = {"feedback_score": True}
            version_metadata_col_names = ["metadata_1"]

            _render_list_tab(
                df,
                feedback_col_names,
                feedback_directions,
                version_metadata_col_names,
            )

        app = AppTestHelper.create_and_run_app(test_app)
        AppTestHelper.assert_no_errors(app)

    def test_leaderboard_main_no_app_selected(self, mock_data):
        """Test main leaderboard function when no app is selected."""
        with MockManager.mock_dashboard_utils(mock_data):
            with patch(
                "trulens.dashboard.utils.dashboard_utils.render_sidebar"
            ) as mock_sidebar:
                mock_sidebar.return_value = None

                def test_app():
                    from trulens.dashboard.tabs.Leaderboard import (
                        leaderboard_main,
                    )

                    leaderboard_main()

                app = AppTestHelper.create_and_run_app(test_app)
                AppTestHelper.assert_no_errors(app)

    def test_leaderboard_main_with_app_selected(self, mock_data):
        """Test main leaderboard function with app selected."""
        with MockManager.mock_all_common_dependencies(mock_data):

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import leaderboard_main

                leaderboard_main()

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_page_components_integration(self, mock_data):
        """Test integration between different page components."""
        with MockManager.mock_all_common_dependencies(mock_data):

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import (
                    render_leaderboard,
                )

                render_leaderboard("Test App")

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)
            assert len(app.divider) > 0


class TestLeaderboardUIErrorHandling:
    """Test error handling in Leaderboard UI components."""

    def test_handle_database_connection_error(self):
        """Test handling of database connection errors."""

        def test_app():
            import streamlit as st
            from trulens.dashboard.tabs.Leaderboard import render_leaderboard

            try:
                with patch(
                    "trulens.dashboard.utils.dashboard_utils.get_session"
                ) as mock_get_session:
                    mock_get_session.side_effect = Exception(
                        "Database connection failed"
                    )
                    render_leaderboard("Test App")
            except Exception:
                st.error("Failed: Database connection failed")

        AppTestHelper.create_and_run_app(test_app, should_raise=True)

    def test_handle_invalid_data_format(self):
        """Test handling of invalid data formats."""

        def test_app():
            import streamlit as st
            from trulens.dashboard.tabs.Leaderboard import render_leaderboard

            try:
                with MockManager.mock_dashboard_utils():
                    with patch(
                        "trulens.dashboard.utils.dashboard_utils.get_records_and_feedback"
                    ) as mock_get_records:
                        mock_get_records.side_effect = Exception(
                            "Invalid data format"
                        )
                        render_leaderboard("Test App")
            except Exception:
                st.error("Failed: Invalid data format")

        AppTestHelper.create_and_run_app(test_app, should_raise=True)

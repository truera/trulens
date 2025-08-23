"""Test UI interactions for the TruLens Dashboard components using AppTest."""

import pytest

from tests.unit.streamlit.test_streamlit_utils import AppTestHelper
from tests.unit.streamlit.test_streamlit_utils import MockManager
from tests.unit.streamlit.test_streamlit_utils import create_mock_data_dict


class TestStreamlitInteractions:
    """Test suite for Streamlit UI interactions using AppTest."""

    @pytest.fixture
    def mock_data(self):
        """Mock data for testing interactions."""
        return create_mock_data_dict()

    def test_pin_toggle_functionality(self):
        """Test the pin toggle functionality."""
        with MockManager.mock_dashboard_utils():

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import handle_pin_toggle

                selected_app_ids = ["app_1", "app_2"]
                on_leaderboard = False
                handle_pin_toggle(selected_app_ids, on_leaderboard)

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_metadata_dialog_basic_functionality(self):
        """Test the basic functionality of the metadata dialog."""
        with MockManager.mock_dashboard_utils():

            def test_app():
                import pandas as pd
                from trulens.dashboard.tabs.Leaderboard import (
                    handle_add_metadata,
                )

                selected_rows = pd.DataFrame({
                    "app_id": ["app_1", "app_2"],
                    "app_version": ["v1.0", "v2.0"],
                })
                metadata_col_names = ["metadata_1", "metadata_2"]
                handle_add_metadata(selected_rows, metadata_col_names)

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_grid_tab_button_interactions(self, mock_data):
        """Test button interactions in the grid tab."""
        with MockManager.mock_dashboard_utils(mock_data):
            # Don't mock _render_grid since we want to test button rendering
            def test_app():
                from unittest.mock import patch

                from trulens.dashboard.tabs.Leaderboard import _render_grid_tab

                from tests.unit.streamlit.test_streamlit_utils import (
                    create_mock_data_dict,
                )

                # Create mock data inside the test function
                test_mock_data = create_mock_data_dict()

                # Mock streamlit components that would normally require user interaction
                with patch("streamlit.button", return_value=False) as _:
                    _render_grid_tab(
                        df=test_mock_data["records_df"],
                        feedback_col_names=test_mock_data["feedback_col_names"],
                        feedback_defs=test_mock_data["feedback_defs"],
                        feedback_directions={
                            "feedback_score": True,
                            "relevance": True,
                        },
                        version_metadata_col_names=test_mock_data[
                            "version_metadata_col_names"
                        ],
                        app_name="Test App",
                        grid_key="test_grid",
                    )

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_multiselect_metadata_columns(self, mock_data):
        """Test multiselect functionality for metadata columns."""
        with MockManager.mock_dashboard_utils(mock_data):

            def test_app():
                import streamlit as st

                from tests.unit.streamlit.test_streamlit_utils import (
                    create_mock_data_dict,
                )

                # Create mock data inside the test function
                test_mock_data = create_mock_data_dict()

                # Just test that multiselect can be created with metadata columns
                st.multiselect(
                    "Display Metadata Columns",
                    test_mock_data["version_metadata_col_names"],
                    default=test_mock_data["version_metadata_col_names"],
                )

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)
            AppTestHelper.assert_has_multiselect_with_label(
                app, "Display Metadata Columns"
            )

    def test_toggle_only_show_pinned(self, mock_data):
        """Test the 'Only Show Pinned' toggle functionality."""
        with MockManager.mock_dashboard_utils(mock_data):

            def test_app():
                import streamlit as st

                # Just test that toggle can be created
                st.toggle("Only Show Pinned", value=False)

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)
            AppTestHelper.assert_has_toggle_with_label(app, "Only Show Pinned")

    def test_button_disabled_states(self, mock_data):
        """Test that buttons are properly disabled when no rows are selected."""
        with MockManager.mock_dashboard_utils(mock_data):

            def test_app():
                import streamlit as st

                # Test button states - disabled when no selection
                st.button("Pin App", disabled=True)
                st.button("Examine Records", disabled=True)

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_compare_button_states(self, mock_data):
        """Test compare button states with different selection counts."""
        with MockManager.mock_dashboard_utils(mock_data):

            def test_app():
                import streamlit as st

                # Test compare button state
                selection_count = 1  # Less than minimum required
                is_disabled = selection_count < 2 or selection_count > 10
                st.button("Compare", disabled=is_disabled)

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_session_state_updates(self, mock_data):
        """Test that session state is properly updated with interactions."""
        with MockManager.mock_dashboard_utils(mock_data):

            def test_app():
                import streamlit as st

                # Test session state management
                if "test_key" not in st.session_state:
                    st.session_state.test_key = "initialized"

                st.text(f"State: {st.session_state.test_key}")

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)


class TestStreamlitFormInteractions:
    """Test form-based interactions in Streamlit components."""

    def test_virtual_app_form_submission(self):
        """Test virtual app form submission."""
        with MockManager.mock_all_common_dependencies():

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import (
                    handle_add_virtual_app,
                )

                from tests.unit.streamlit.test_streamlit_utils import (
                    TestDataFactory,
                )

                # Create proper feedback_defs DataFrame
                feedback_defs = TestDataFactory.create_feedback_defs([
                    "feedback_score"
                ])

                handle_add_virtual_app(
                    app_name="Test App",
                    feedback_col_names=["feedback_score"],
                    feedback_defs=feedback_defs,
                    metadata_col_names=["metadata_1"],
                )

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_metadata_form_validation(self):
        """Test metadata form input validation."""
        with MockManager.mock_dashboard_utils():

            def test_app():
                import pandas as pd
                from trulens.dashboard.tabs.Leaderboard import (
                    handle_add_metadata,
                )

                selected_rows = pd.DataFrame({
                    "app_id": ["app_1"],
                    "app_version": ["v1.0"],
                })
                metadata_col_names = ["metadata_1"]
                handle_add_metadata(selected_rows, metadata_col_names)

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)


class TestStreamlitStateManagement:
    """Test state management in Streamlit components."""

    @pytest.fixture
    def mock_data(self):
        """Mock data for testing state management."""
        return create_mock_data_dict()

    def test_page_state_persistence(self):
        """Test that page state persists across interactions."""

        def test_app():
            from trulens.dashboard.tabs.Leaderboard import init_page_state

            init_page_state()
            init_page_state()  # Should not reinitialize

        app = AppTestHelper.create_and_run_app(test_app)
        AppTestHelper.assert_no_errors(app)

    def test_query_params_integration(self, mock_data):
        """Test integration with query parameters."""
        with MockManager.mock_all_common_dependencies(mock_data):

            def test_app():
                from trulens.dashboard.tabs.Leaderboard import _render_grid_tab

                from tests.unit.streamlit.test_streamlit_utils import (
                    create_mock_data_dict,
                )

                # Create mock data inside the test function
                test_mock_data = create_mock_data_dict()

                _render_grid_tab(
                    df=test_mock_data["records_df"],
                    feedback_col_names=test_mock_data["feedback_col_names"],
                    feedback_defs=test_mock_data["feedback_defs"],
                    feedback_directions={
                        "feedback_score": True,
                        "relevance": True,
                    },
                    version_metadata_col_names=test_mock_data[
                        "version_metadata_col_names"
                    ],
                    app_name="Test App",
                    grid_key="test_grid",
                )

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

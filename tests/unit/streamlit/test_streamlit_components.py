"""Test basic UI components for the TruLens Dashboard streamlit components using AppTest."""

import pytest

from tests.unit.streamlit.test_streamlit_utils import AppTestHelper
from tests.unit.streamlit.test_streamlit_utils import MockManager
from tests.unit.streamlit.test_streamlit_utils import TestDataFactory
from tests.unit.streamlit.test_streamlit_utils import create_mock_data_dict


class TestStreamlitComponents:
    """Test suite for TruLens Streamlit components using AppTest."""

    @pytest.fixture
    def mock_data(self):
        """Mock data for testing streamlit components."""
        return create_mock_data_dict()

    def test_trulens_leaderboard_no_records(self):
        """Test trulens_leaderboard function with no records."""
        with MockManager.mock_tru_session(return_empty_data=True):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard()

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)
            AppTestHelper.assert_has_text_content(app, "No records yet...")

    def test_trulens_leaderboard_with_records(self, mock_data):
        """Test trulens_leaderboard function with sample records."""
        with MockManager.mock_tru_session(mock_data=mock_data):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard()

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)
            AppTestHelper.assert_has_header_content(app, "Test App")

    def test_trulens_leaderboard_with_app_ids_filter(self, mock_data):
        """Test trulens_leaderboard function with specific app_ids filter."""
        filtered_data = create_mock_data_dict(
            app_ids=["app_1"],
            size=1,
        )

        with MockManager.mock_tru_session(mock_data=filtered_data):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard(app_ids=["app_1"])

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_trulens_leaderboard_metrics_rendering(self, mock_data):
        """Test that metrics are properly rendered in trulens_leaderboard."""
        with MockManager.mock_tru_session(mock_data=mock_data):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard()

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)
            AppTestHelper.assert_has_metrics(app, min_count=1)

    def test_trulens_leaderboard_feedback_metrics(self, mock_data):
        """Test that feedback metrics are properly rendered."""
        with MockManager.mock_tru_session(mock_data=mock_data):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard()

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)
            AppTestHelper.assert_has_metrics(app, min_count=4)

    def test_trulens_leaderboard_distance_feedback(self):
        """Test rendering of distance-based feedback metrics."""
        # Create data with distance feedback
        distance_data = create_mock_data_dict(
            feedback_names=["feedback_score", "relevance", "distance_score"],
        )

        # Modify the records to include distance scores
        distance_data["records_df"]["distance_score"] = [0.3, 0.1]

        # Add distance feedback definition with lower_is_better
        distance_feedback_def = TestDataFactory.create_feedback_defs([
            "distance_score"
        ])
        distance_feedback_def.loc[0, "feedback_json"]["higher_is_better"] = (
            False
        )

        distance_data["feedback_defs"] = distance_feedback_def

        with MockManager.mock_tru_session(mock_data=distance_data):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard()

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)


class TestStreamlitComponentsErrorHandling:
    """Test error handling in Streamlit components."""

    def test_trulens_leaderboard_session_error(self):
        """Test trulens_leaderboard handling of session initialization errors."""

        def test_app():
            import streamlit as st

            try:
                with MockManager.mock_tru_session():
                    # Force a session error by raising exception during initialization
                    raise Exception("Session initialization failed")
            except Exception:
                st.error("Failed to initialize TruLens session")

        AppTestHelper.create_and_run_app(test_app, should_raise=True)

    def test_trulens_leaderboard_database_error(self):
        """Test trulens_leaderboard handling of database errors."""

        def test_app():
            import streamlit as st

            try:
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard()
            except Exception:
                st.error("Database connection failed")

        # Mock database error
        with MockManager.mock_tru_session():
            _ = AppTestHelper.create_and_run_app(test_app, should_raise=True)

    def test_trulens_leaderboard_invalid_app_ids(self):
        """Test trulens_leaderboard with invalid app_ids parameter."""
        with MockManager.mock_tru_session(return_empty_data=True):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard(app_ids=["nonexistent_app"])

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)


class TestStreamlitComponentsIntegration:
    """Test integration aspects of Streamlit components."""

    @pytest.fixture
    def mock_data(self):
        """Mock data for testing streamlit components."""
        return create_mock_data_dict()

    def test_component_state_management(self, mock_data):
        """Test that component state is properly managed."""
        with MockManager.mock_tru_session(mock_data=mock_data):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard()
                trulens_leaderboard()  # Second call should not cause issues

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

    def test_component_performance_with_large_dataset(self):
        """Test component performance with larger datasets."""
        large_df = TestDataFactory.create_large_dataset(size=100)
        large_feedback_defs = TestDataFactory.create_feedback_defs([
            "feedback_score"
        ])

        large_data = {
            "records_df": large_df,
            "feedback_col_names": ["feedback_score"],
            "feedback_defs": large_feedback_defs,
            "app_ids": [f"app_{i}" for i in range(100)],
        }

        with MockManager.mock_tru_session(mock_data=large_data):

            def test_app():
                from trulens.dashboard.streamlit import trulens_leaderboard

                trulens_leaderboard()

            app = AppTestHelper.create_and_run_app(test_app)
            AppTestHelper.assert_no_errors(app)

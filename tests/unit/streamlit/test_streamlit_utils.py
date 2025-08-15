"""Shared utilities for Streamlit UI tests."""

from contextlib import contextmanager
from typing import Callable, List, Optional
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class TestDataFactory:
    """Factory for creating test data across all Streamlit tests."""

    @staticmethod
    def create_versions_df(app_ids=None, versions=None, names=None):
        """Create a mock versions dataframe."""
        app_ids = app_ids or ["app_1", "app_2"]
        size = len(app_ids)
        versions = versions or [f"v{i + 1}.0" for i in range(size)]
        names = names or ["Test App"] * size

        return pd.DataFrame({
            "app_id": app_ids,
            "app_version": versions,
            "app_name": names,
        })

    @staticmethod
    def create_records_df(app_ids=None, size=2, feedback_names=None):
        """Create a mock records dataframe."""
        app_ids = app_ids or ["app_1", "app_2"]
        feedback_names = feedback_names or ["feedback_score", "relevance"]
        size = len(app_ids)

        # Base DataFrame structure
        base_data = {
            "app_id": app_ids,
            "app_name": ["Test App"] * size,
            "app_version": [f"v{i + 1}.0" for i in range(size)],
            "record_id": [f"record_{i + 1}" for i in range(size)],
            "input": [f"Input {i + 1}" for i in range(size)],
            "output": [f"Output {i + 1}" for i in range(size)],
            "cost_currency": ["USD"] * size,
            "cost": [0.01 * (i + 1) for i in range(size)],
            "total_cost": [0.01 * (i + 1) for i in range(size)],
            "latency": [0.5 + 0.1 * i for i in range(size)],
            "Average Latency (s)": [0.5 + 0.1 * i for i in range(size)],
            "Records": [10 + i for i in range(size)],
            "total_tokens": [100 + 50 * i for i in range(size)],
            "ts": [
                pd.Timestamp.now() - pd.Timedelta(days=i) for i in range(size)
            ],
            "tags": [f"tag_{i + 1}" for i in range(size)],
            "record_json": [{"key": f"value_{i + 1}"} for i in range(size)],
            "metadata_1": [f"meta1_value_{i + 1}" for i in range(size)],
            "metadata_2": [f"meta2_value_{i + 1}" for i in range(size)],
            "app_json": [
                {"app_name": f"Test App {i + 1}", "app_version": f"v{i + 1}.0"}
                for i in range(size)
            ],
        }

        # Add feedback columns
        for feedback_name in feedback_names:
            if "distance" in feedback_name:
                # Distance metrics typically have lower values being better
                base_data[feedback_name] = [0.1 + 0.1 * i for i in range(size)]
            else:
                # Standard metrics where higher is better
                base_data[feedback_name] = [0.8 + 0.1 * i for i in range(size)]

        return pd.DataFrame(base_data)

    @staticmethod
    def create_processed_df(
        app_ids=None, include_feedback=True, include_metadata=True
    ):
        """Create a processed dataframe for leaderboard display."""
        app_ids = app_ids or ["app_1", "app_2"]
        size = len(app_ids)

        data = {
            "app_version": [f"v{i + 1}.0" for i in range(size)],
            "app_id": app_ids,
            "app_name": ["Test App"] * size,
            "Records": [10 + i * 5 for i in range(size)],
            "Average Latency (s)": [0.5 + i * 0.2 for i in range(size)],
            "tags": [[] for _ in range(size)],
            "Total Cost (Snowflake Credits)": [0.0] * size,
            "Total Cost (USD)": [0.01 + i * 0.01 for i in range(size)],
            "Total Tokens": [100 + i * 50 for i in range(size)],
        }

        if include_feedback:
            data.update({
                "feedback_score": [0.8 + i * 0.1 for i in range(size)],
                "relevance": [0.7 + i * 0.15 for i in range(size)],
            })

        if include_metadata:
            data.update({
                "metadata_1": [f"meta{i + 1}" for i in range(size)],
                "metadata_2": [f"meta{i + 3}" for i in range(size)],
            })

        return pd.DataFrame(data)

    @staticmethod
    def create_feedback_defs(feedback_names=None):
        """Create mock feedback definitions."""
        feedback_names = feedback_names or ["feedback_score", "relevance"]

        return pd.DataFrame({
            "feedback_name": feedback_names,
            "feedback_json": [
                {
                    "supplied_name": name,
                    "implementation": {"name": name},
                    "higher_is_better": True,
                }
                for name in feedback_names
            ],
        })

    @staticmethod
    def create_large_dataset(size=100):
        """Create a large dataset for performance testing."""
        return pd.DataFrame({
            "app_id": [f"app_{i}" for i in range(size)],
            "app_json": [
                {
                    "app_name": f"Test App {i}",
                    "app_version": f"v{i}.0",
                    "metadata": {"environment": "test"},
                    "tags": ["test"],
                }
                for i in range(size)
            ],
            "latency": [0.5 + i * 0.01 for i in range(size)],
            "total_cost": [0.01 + i * 0.001 for i in range(size)],
            "total_tokens": [100 + i * 10 for i in range(size)],
            "feedback_score": [0.5 + (i % 50) * 0.01 for i in range(size)],
        })


class AppTestHelper:
    """Helper class for common AppTest operations."""

    @staticmethod
    def create_and_run_app(
        test_function: Callable, should_raise=False, timeout=10
    ) -> AppTest:
        """Create and run an AppTest from a function."""
        app = AppTest.from_function(test_function)
        app.run(timeout=timeout)

        if not should_raise:
            assert not app.exception, f"Unexpected exception: {app.exception}"

        return app

    @staticmethod
    def assert_no_errors(app: AppTest):
        """Assert that the app ran without errors."""
        assert not app.exception

    @staticmethod
    def assert_has_title_content(app: AppTest, expected_content: str):
        """Assert that the app has title content."""
        assert len(app.title) > 0
        assert expected_content in app.title[0].value

    @staticmethod
    def assert_has_header_content(app: AppTest, expected_content: str):
        """Assert that the app has header content."""
        headers_found = False
        for header in app.header:
            if expected_content in str(header.value):
                headers_found = True
                break
        assert (
            headers_found
        ), f"Header with '{expected_content}' should be present"

    @staticmethod
    def assert_has_error_with_message(app: AppTest, expected_message: str):
        """Assert that the app has an error with specific message."""
        assert len(app.error) > 0
        assert expected_message in str(app.error[0].value)

    @staticmethod
    def assert_has_warning_with_message(app: AppTest, expected_message: str):
        """Assert that the app has a warning with specific message."""
        assert len(app.warning) > 0
        assert expected_message in app.warning[0].value

    @staticmethod
    def assert_has_text_content(app: AppTest, expected_text: str):
        """Assert that the app contains specific text content."""
        all_text = ""
        # Convert ElementLists to regular lists for concatenation
        text_elements = list(app.markdown) + list(app.text)
        for element in text_elements:
            all_text += str(
                element.value if hasattr(element, "value") else element
            )
        assert (
            expected_text in all_text
        ), f"Expected text '{expected_text}' not found in app content"

    @staticmethod
    def assert_has_metrics(app: AppTest, min_count=1):
        """Assert that the app has rendered metrics."""
        assert (
            len(app.metric) >= min_count
        ), f"Should have at least {min_count} metrics"

    @staticmethod
    def assert_has_buttons(
        app: AppTest, expected_labels: Optional[List[str]] = None
    ):
        """Assert that the app has buttons with expected labels."""
        assert len(app.button) > 0, "Should have action buttons"

        if expected_labels:
            button_labels = [button.label for button in app.button]
            for label in expected_labels:
                assert any(
                    label in btn_label for btn_label in button_labels
                ), f"Button with label '{label}' should be present"

    @staticmethod
    def assert_has_multiselect_with_label(app: AppTest, expected_label: str):
        """Assert that the app has a multiselect with specific label."""
        multiselect_found = False
        for multiselect in app.multiselect:
            if expected_label in str(multiselect.label):
                multiselect_found = True
                break
        assert (
            multiselect_found
        ), f"Multiselect with label '{expected_label}' should be present"

    @staticmethod
    def assert_has_toggle_with_label(app: AppTest, expected_label: str):
        """Assert that the app has a toggle with specific label."""
        toggle_found = False
        for toggle in app.toggle:
            if expected_label in str(toggle.label):
                toggle_found = True
                break
        assert (
            toggle_found
        ), f"Toggle with label '{expected_label}' should be present"


class MockManager:
    """Centralized mock management for tests."""

    @staticmethod
    @contextmanager
    def mock_tru_session(return_empty_data=False, mock_data=None):
        """Context manager for mocking TruSession."""
        with patch("trulens.core.session.TruSession") as mock_tru_session:
            mock_session = Mock()
            mock_db = Mock()

            if return_empty_data:
                mock_db.get_records_and_feedback.return_value = (
                    pd.DataFrame(),
                    [],
                )
                mock_db.get_feedback_defs.return_value = pd.DataFrame()
                mock_db.get_apps.return_value = []
            elif mock_data:
                mock_db.get_records_and_feedback.return_value = (
                    mock_data.get(
                        "records_df", TestDataFactory.create_records_df()
                    ),
                    mock_data.get(
                        "feedback_col_names", ["feedback_score", "relevance"]
                    ),
                )
                mock_db.get_feedback_defs.return_value = mock_data.get(
                    "feedback_defs", TestDataFactory.create_feedback_defs()
                )
                mock_db.get_apps.return_value = mock_data.get(
                    "app_ids", ["app_1", "app_2"]
                )
            else:
                # Default mock data
                mock_db.get_records_and_feedback.return_value = (
                    TestDataFactory.create_records_df(),
                    ["feedback_score", "relevance"],
                )
                mock_db.get_feedback_defs.return_value = (
                    TestDataFactory.create_feedback_defs()
                )
                mock_db.get_apps.return_value = ["app_1", "app_2"]

            mock_session.connector.db = mock_db
            mock_session.experimental_feature = Mock(return_value=False)
            mock_tru_session.return_value = mock_session
            yield mock_session

    @staticmethod
    @contextmanager
    def mock_dashboard_utils(
        mock_data=None, empty_versions=False, empty_records=False
    ):
        """Context manager for mocking dashboard utilities."""
        if mock_data is None:
            mock_data = {
                "versions_df": TestDataFactory.create_versions_df(),
                "records_df": TestDataFactory.create_records_df(),
                "feedback_col_names": ["feedback_score", "relevance"],
                "version_metadata_col_names": ["metadata_1", "metadata_2"],
            }

        versions_df = (
            pd.DataFrame() if empty_versions else mock_data["versions_df"]
        )
        records_df = (
            pd.DataFrame() if empty_records else mock_data["records_df"]
        )

        with patch.multiple(
            "trulens.dashboard.utils.dashboard_utils",
            set_page_config=Mock(),
            render_sidebar=Mock(return_value="Test App"),
            render_app_version_filters=Mock(
                return_value=(
                    versions_df,
                    mock_data["version_metadata_col_names"],
                )
            ),
            get_records_and_feedback=Mock(
                return_value=(records_df, mock_data["feedback_col_names"])
            ),
            get_feedback_defs=Mock(return_value=({}, {})),
            get_session=Mock(),
            is_sis_compatibility_enabled=Mock(return_value=False),
            _show_no_records_error=Mock(),
            _check_cross_format_records=Mock(return_value=(0, 0)),
            ST_RECORDS_LIMIT="records_limit",
            update_app_metadata=Mock(),
            get_app_versions=Mock(),
            get_apps=Mock(),
            read_query_params_into_session_state=Mock(),
        ):
            yield mock_data

    @staticmethod
    @contextmanager
    def mock_leaderboard_components():
        """Context manager for mocking leaderboard rendering components."""
        with patch.multiple(
            "trulens.dashboard.tabs.Leaderboard",
            _render_plot_tab=Mock(return_value=None),
            _render_list_tab=Mock(return_value=None),
            _render_grid=Mock(return_value=pd.DataFrame()),
            _render_grid_tab=Mock(return_value=None),
            _preprocess_df=Mock(
                return_value=TestDataFactory.create_processed_df()
            ),
            virtual_app=Mock(),
        ):
            yield

    @staticmethod
    @contextmanager
    def mock_otel_tracing():
        """Context manager for mocking OTEL tracing."""
        with patch(
            "trulens.core.otel.utils.is_otel_tracing_enabled",
            return_value=False,
        ):
            yield

    @staticmethod
    @contextmanager
    def mock_all_common_dependencies(mock_data=None, empty_data=False):
        """Context manager that mocks all common dependencies at once."""
        with MockManager.mock_dashboard_utils(
            mock_data, empty_records=empty_data, empty_versions=empty_data
        ):
            with MockManager.mock_leaderboard_components():
                with MockManager.mock_otel_tracing():
                    yield


def create_mock_data_dict(app_ids=None, feedback_names=None, size=2):
    """Create a complete mock data dictionary for tests."""
    app_ids = app_ids or ["app_1", "app_2"]
    feedback_names = feedback_names or ["feedback_score", "relevance"]

    return {
        "records_df": TestDataFactory.create_records_df(
            app_ids=app_ids, size=size, feedback_names=feedback_names
        ),
        "feedback_col_names": feedback_names,
        "feedback_defs": TestDataFactory.create_feedback_defs(feedback_names),
        "versions_df": TestDataFactory.create_versions_df(app_ids=app_ids),
        "version_metadata_col_names": ["metadata_1", "metadata_2"],
        "app_ids": app_ids,
    }

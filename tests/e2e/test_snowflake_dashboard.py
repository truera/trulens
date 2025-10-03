from typing import List
from unittest.mock import Mock
from unittest.mock import patch

from snowflake.connector.errors import DatabaseError
from trulens.dashboard import run_dashboard
from trulens.dashboard.main import main

from tests.util.otel_test_case import OtelTestCase
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestOtelSnowflakeDashboard(OtelTestCase, SnowflakeTestCase):
    def _test_run_dashboard(
        self,
        mock_popen: Mock,
        spcs_mode: bool,
        expected_snowflake_args: List[str],
    ) -> None:
        # Mock the subprocess.Popen to prevent actual dashboard execution.
        mock_proc = Mock()
        # Simulate the process running for two checks, then ending successfully.
        mock_proc.poll.side_effect = [None, None] + [0] * 1000
        mock_proc.stdout.readline.return_value = (
            "Local URL: http://localhost:8501"
        )
        mock_proc.stderr.readline.return_value = ""
        mock_popen.return_value = mock_proc
        # Call `run_dashboard`
        result = run_dashboard(spcs_mode=spcs_mode)
        # Verify.
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        popen_args = call_args[0][0]  # First positional argument is the list
        self.assertEqual("streamlit", popen_args[0])
        self.assertEqual("run", popen_args[1])
        snowflake_args = popen_args[popen_args.index("--") + 3 :]
        self.assertListEqual(expected_snowflake_args, snowflake_args)
        self.assertEqual(result, mock_proc)

    @patch("trulens.dashboard.run.subprocess.Popen")
    def test_run_dashboard_non_spcs_mode(self, mock_popen: Mock) -> None:
        tru_session = self.get_session("test_run_dashboard_non_spcs")
        snowpark_session = tru_session.connector.snowpark_session
        self._test_run_dashboard(
            mock_popen,
            spcs_mode=False,
            expected_snowflake_args=[
                "--snowflake-account",
                snowpark_session.get_current_account()[1:-1],
                "--snowflake-user",
                snowpark_session.get_current_user()[1:-1],
                "--snowflake-role",
                snowpark_session.get_current_role()[1:-1],
                "--snowflake-database",
                snowpark_session.get_current_database()[1:-1],
                "--snowflake-schema",
                snowpark_session.get_current_schema()[1:-1],
                "--snowflake-warehouse",
                snowpark_session.get_current_warehouse()[1:-1],
                "--snowflake-host",
                snowpark_session.connection.host,
                "--snowflake-authenticator",
                "externalbrowser",
                "--snowflake-use-account-event-table",
            ],
        )

    @patch("trulens.dashboard.run.subprocess.Popen")
    def test_run_dashboard_in_spcs_mode(self, mock_popen: Mock) -> None:
        tru_session = self.get_session("test_run_dashboard_spcs")
        snowpark_session = tru_session.connector.snowpark_session
        self._test_run_dashboard(
            mock_popen,
            spcs_mode=True,
            expected_snowflake_args=[
                "--snowflake-spcs-mode",
                "--snowflake-account",
                snowpark_session.get_current_account()[1:-1],
                "--snowflake-user",
                snowpark_session.get_current_user()[1:-1],
                "--snowflake-role",
                snowpark_session.get_current_role()[1:-1],
                "--snowflake-database",
                snowpark_session.get_current_database()[1:-1],
                "--snowflake-schema",
                snowpark_session.get_current_schema()[1:-1],
                "--snowflake-warehouse",
                snowpark_session.get_current_warehouse()[1:-1],
                "--snowflake-host",
                snowpark_session.connection.host,
                "--snowflake-use-account-event-table",
            ],
        )

    @patch("trulens.dashboard.utils.dashboard_utils.read_spcs_oauth_token")
    @patch(
        "trulens.dashboard.utils.dashboard_utils.argparse.ArgumentParser.parse_args"
    )
    def test_main_in_spcs_mode(
        self, mock_parse_args: Mock, mock_read_oauth_token: Mock
    ) -> None:
        # Get session to extract Snowflake connection parameters.
        tru_session = self.get_session("test_run_in_spcs_mode")
        snowpark_session = tru_session.connector.snowpark_session
        # Mock the OAuth token reading function.
        mock_read_oauth_token.return_value = "fake_oauth_token"
        # Create a mock args object with the required attributes for SPCS mode.
        mock_args = Mock()
        mock_args.database_url = None
        mock_args.snowflake_account = snowpark_session.get_current_account()[
            1:-1
        ]
        mock_args.snowflake_user = snowpark_session.get_current_user()[1:-1]
        mock_args.snowflake_role = snowpark_session.get_current_role()[1:-1]
        mock_args.snowflake_database = snowpark_session.get_current_database()[
            1:-1
        ]
        mock_args.snowflake_schema = snowpark_session.get_current_schema()[1:-1]
        mock_args.snowflake_warehouse = (
            snowpark_session.get_current_warehouse()[1:-1]
        )
        mock_args.snowflake_password = None
        mock_args.snowflake_authenticator = None
        mock_args.snowflake_host = snowpark_session.connection.host
        mock_args.snowflake_spcs_mode = True
        mock_args.snowflake_use_account_event_table = True
        mock_args.sis_compatibility = False
        mock_args.database_prefix = "trulens_"
        mock_args.otel_tracing = True
        mock_parse_args.return_value = mock_args
        with self.assertRaisesRegex(
            DatabaseError, "Invalid OAuth access token"
        ):
            main()
        mock_read_oauth_token.assert_called_once()

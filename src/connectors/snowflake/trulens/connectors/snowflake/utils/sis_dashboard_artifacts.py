import glob
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple, Union

from trulens.connectors.snowflake.utils.server_side_evaluation_artifacts import (
    _STAGE_NAME as _PKG_STAGE_NAME,
)

from snowflake.connector.errors import ProgrammingError
from snowflake.snowpark import Session

_STAGE_NAME = "TRULENS_DASHBOARD_STAGE"
_STREAMLIT_ENTRYPOINT = "Leaderboard.py"

_TRULENS_DEPENDENCIES = [
    "trulens-core",
    "trulens-dashboard",
    "trulens-connectors-snowflake",
]


class SiSDashboardArtifacts:
    """This class is used to set up Snowflake artifacts for launching the dashboard on SiS."""

    def __init__(
        self,
        streamlit_name: str,
        session: Session,
        database: str,
        schema: str,
        warehouse: str,
        use_staged_packages: bool,
    ) -> None:
        self._validate_streamlit_name(streamlit_name)
        self._streamlit_name = streamlit_name
        self._session = session
        self._database = database
        self._schema = schema
        self._warehouse = warehouse
        self._use_staged_packages = use_staged_packages

    def set_up_all(self) -> None:
        self._set_up_stage()
        return self._set_up_streamlit()

    def _run_query(self, q: str) -> Union[List[Tuple], List[Dict]]:
        cursor = self._session.connection.cursor()
        cursor.execute(q)
        return cursor.fetchall()

    def _stage_file(
        self, file_path: str, stage_path: Optional[str] = None
    ) -> None:
        if not stage_path:
            full_stage_path = _STAGE_NAME
        else:
            full_stage_path = f"{_STAGE_NAME}/{stage_path}"

        self._run_query(
            f"PUT file://{file_path} @{full_stage_path} OVERWRITE = TRUE AUTO_COMPRESS = FALSE"
        )

    def _validate_streamlit_name(self, streamlit_name: str):
        if not streamlit_name:
            raise ValueError("`streamlit_name` cannot be empty!")
        if not re.match(r"^[A-Za-z0-9_]+$", streamlit_name):
            raise ValueError(
                "`streamlit_name` must contain only alphanumeric and underscore characters!"
            )

    def _set_up_environment_file(self, environment_filepath: str) -> None:
        if self._use_staged_packages:
            self._stage_file(environment_filepath)
        else:
            with open(environment_filepath, "r") as env_f:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    new_env_path = os.path.join(tmp_dir, "environment.yml")
                    with open(new_env_path, "w") as f:
                        f.write(env_f.read())
                        for dep in _TRULENS_DEPENDENCIES:
                            f.write(f"- {dep}\n")
                        f.flush()
                        self._stage_file(new_env_path)

    def _set_up_stage(self) -> None:
        self._run_query(f"CREATE STAGE IF NOT EXISTS {_STAGE_NAME}")
        data_directory = os.path.join(
            os.path.dirname(__file__), "../../../data/sis_dashboard"
        )

        # Stage the environment file
        self._set_up_environment_file(
            os.path.join(data_directory, "environment.yml")
        )

        # Stage the main dashboard file
        entrypoint_path = os.path.join(data_directory, _STREAMLIT_ENTRYPOINT)
        if not os.path.exists(entrypoint_path) or not os.path.isfile(
            entrypoint_path
        ):
            raise ValueError(
                f"Main dashboard file '{entrypoint_path}' does not exist."
            )

        self._stage_file(entrypoint_path)

        # Stage the remaining pages
        for pagefile in glob.glob(
            os.path.join(data_directory, "pages", "*.py")
        ):
            file_path = os.path.join(data_directory, "pages", pagefile)
            self._stage_file(file_path, "pages")

    def _set_up_streamlit(self) -> None:
        if self._use_staged_packages:
            imports = f"""
            IMPORTS = (
                    "@{self._database}.{self._schema}.{_PKG_STAGE_NAME}/trulens-core.zip",
                    "@{self._database}.{self._schema}.{_PKG_STAGE_NAME}/trulens-dashboard.zip",
                    "@{self._database}.{self._schema}.{_PKG_STAGE_NAME}/trulens-connectors-snowflake.zip"
                )
            """
        else:
            imports = ""
        try:
            return self._run_query(
                f"""
                CREATE STREAMLIT IF NOT EXISTS {self._streamlit_name}
                FROM @{self._database}.{self._schema}.{_STAGE_NAME}
                MAIN_FILE = "{_STREAMLIT_ENTRYPOINT}"
                QUERY_WAREHOUSE = "{self._warehouse}"
                TITLE = "{self._streamlit_name}"
                {imports}
                """
            )[0][0]
        except ProgrammingError:
            return self._run_query(
                f"""
                CREATE STREAMLIT IF NOT EXISTS {self._streamlit_name}
                ROOT_LOCATION=@{self._database}.{self._schema}.{_STAGE_NAME}
                MAIN_FILE = "{_STREAMLIT_ENTRYPOINT}"
                QUERY_WAREHOUSE = "{self._warehouse}"
                TITLE = "{self._streamlit_name}"
                {imports}
                """
            )[0][0]

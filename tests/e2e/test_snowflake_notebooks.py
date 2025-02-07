import tempfile
from typing import Sequence

from trulens.connectors.snowflake.utils.server_side_evaluation_artifacts import (
    _TRULENS_PACKAGES,
)
from trulens.connectors.snowflake.utils.server_side_evaluation_artifacts import (
    _TRULENS_PACKAGES_DEPENDENCIES,
)

from tests.util.snowflake_test_case import SnowflakeTestCase

_STAGE_NAME = "SNOWFLAKE_NOTEBOOKS"
_DATA_DIRECTORY = "tests/e2e/data/"


class TestSnowflakeNotebooks(SnowflakeTestCase):
    def test_simple(self) -> None:
        self.create_and_use_schema("test_simple", append_uuid=True)
        self._upload_and_run_notebook("simple", _TRULENS_PACKAGES)

    def test_staged_packages(self) -> None:
        self.get_session("test_staged_packages")
        self._upload_and_run_notebook(
            "staged_packages", _TRULENS_PACKAGES_DEPENDENCIES
        )

    def test_staged_packages_with_otel(self) -> None:
        self.get_session("test_staged_packages_with_otel")
        self._upload_and_run_notebook(
            "staged_packages_with_otel", _TRULENS_PACKAGES_DEPENDENCIES
        )

    def _upload_and_run_notebook(
        self,
        name: str,
        conda_dependencies: Sequence[str],
    ) -> None:
        tmp_directory = tempfile.TemporaryDirectory()
        try:
            self.run_query(f"CREATE STAGE {_STAGE_NAME}")
            self.run_query(
                f"PUT file://{_DATA_DIRECTORY}/{name}.ipynb @{_STAGE_NAME} AUTO_COMPRESS = FALSE"
            )
            self._create_and_upload_environment_yml(
                conda_dependencies, tmp_directory.name
            )
            self.run_query(
                f"""
                CREATE NOTEBOOK {name}
                FROM '@{_STAGE_NAME}'
                MAIN_FILE = '{name}.ipynb'
                QUERY_WAREHOUSE = {self._snowflake_connection_parameters["warehouse"]}
                """
            )
            self.run_query(f"ALTER NOTEBOOK {name} ADD LIVE VERSION FROM LAST")
            self.run_query(f"EXECUTE NOTEBOOK {name}()")
        finally:
            tmp_directory.cleanup()
            self.run_query(f"DROP STAGE IF EXISTS {_STAGE_NAME}")
            self.run_query(f"DROP NOTEBOOK IF EXISTS {name}")

    def _create_and_upload_environment_yml(
        self, conda_dependencies: Sequence[str], directory: str
    ) -> None:
        filename = f"{directory}/environment.yml"
        with open(filename, "w") as fh:
            fh.write("name: app_environment\n")
            fh.write("channels:\n")
            fh.write("- snowflake\n")
            fh.write("dependencies:\n")
            for curr in conda_dependencies:
                fh.write(f"  - {curr}=*\n")
        self.run_query(
            f"PUT file://{filename} @{_STAGE_NAME} AUTO_COMPRESS = FALSE"
        )

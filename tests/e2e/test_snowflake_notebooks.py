from typing import Optional
from unittest import main
import uuid

from tests.util.snowflake_test_case import SnowflakeTestCase

_STAGE_NAME = "SNOWFLAKE_NOTEBOOKS"


class TestSnowflakeNotebooks(SnowflakeTestCase):
    def test_simple(self) -> None:
        self._upload_and_run_notebook(
            "test_simple",
            "simple",
            "tests/e2e/data/",
            "tests/e2e/data/simple_environment",
        )

    def test_staged_packages(self) -> None:
        self.get_session("test_staged_packages")
        self._upload_and_run_notebook(
            None,
            "staged_packages",
            "tests/e2e/data/",
            # TODO(this_pr): Clean up how the environment.yml are generated.
            "tests/e2e/data/staged_packages_environment",
        )

    def _upload_and_run_notebook(
        self,
        schema_base_name: Optional[str],
        name: str,
        path: str,
        environment_yml_path: str,
    ) -> None:
        try:
            if schema_base_name:
                schema_name = (
                    f"{schema_base_name}_{str(uuid.uuid4()).replace('-', '_')}"
                )
                self.create_and_use_schema(schema_name)
            self.run_query(f"CREATE STAGE {_STAGE_NAME}")
            self.run_query(
                f"PUT file://{path}/{name}.ipynb @{_STAGE_NAME} AUTO_COMPRESS = FALSE"
            )
            self.run_query(
                f"PUT file://{environment_yml_path}/environment.yml @{_STAGE_NAME} AUTO_COMPRESS = FALSE"
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
            self.run_query(f"DROP STAGE IF EXISTS {_STAGE_NAME}")
            self.run_query(f"DROP NOTEBOOK IF EXISTS {name}")


if __name__ == "__main__":
    main()

from unittest import main
import uuid

from tests.util.snowflake_test_case import SnowflakeTestCase

_STAGE_NAME = "SNOWFLAKE_NOTEBOOKS"


class TestSnowflakeNotebooks(SnowflakeTestCase):
    def test_simple_notebook(self) -> None:
        self._upload_and_run_notebook(
            "test_simple_notebook",
            "simple",
            "tests/e2e/data/",
        )

    def _upload_and_run_notebook(
        self, schema_base_name: str, name: str, path: str
    ) -> None:
        try:
            schema_name = (
                f"{schema_base_name}_{str(uuid.uuid4()).replace('-', '_')}"
            )
            self.create_and_use_schema(schema_name)
            self.run_query(f"CREATE STAGE {_STAGE_NAME}")
            self.run_query(
                f"PUT file://{path}/{name}.ipynb @{_STAGE_NAME} AUTO_COMPRESS = FALSE"
            )
            self.run_query(
                f"PUT file://{path}/environment.yml @{_STAGE_NAME} AUTO_COMPRESS = FALSE"
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

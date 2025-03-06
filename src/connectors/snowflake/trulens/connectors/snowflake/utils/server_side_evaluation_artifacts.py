import os

from snowflake.snowpark import Session

_STAGE_NAME = "TRULENS_PACKAGES_STAGE"
_STREAM_NAME = "TRULENS_FEEDBACK_EVALS_STREAM"
_STORED_PROCEDURE_NAME = "TRULENS_RUN_DEFERRED_FEEDBACKS"
_WRAPPER_STORED_PROCEDURE_NAME = "TRULENS_RUN_DEFERRED_FEEDBACKS_WRAPPER"
_TASK_NAME = "TRULENS_FEEDBACK_EVALS_TASK"

_PYTHON_STORED_PROCEDURE_CODE_FILENAME = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "server_side_evaluation_stored_procedure.py",
)

_TRULENS_PACKAGES = [
    "trulens-connectors-snowflake",
    "trulens-core",
    "trulens-feedback",
    "trulens-otel-semconv",
    "trulens-providers-cortex",
]

_TRULENS_EXTRA_STAGED_PACKAGES = [
    "trulens-dashboard",
]


# TODO(dkurokawa): get these package versions automatically.
_TRULENS_PACKAGES_DEPENDENCIES = [
    "alembic",
    "dill",
    "munch",
    "nest-asyncio",
    "nltk",
    "numpy",
    "opentelemetry-api",
    "opentelemetry-proto",
    "opentelemetry-sdk",
    "packaging",
    "pandas",
    "pip",
    "pydantic",
    "python-dotenv",
    "requests",
    "rich",
    "scikit-learn",
    "scipy",
    "snowflake-ml-python",
    "snowflake-snowpark-python",
    "snowflake-sqlalchemy",
    "sqlalchemy",
    "tqdm",
    "typing_extensions",
]


class ServerSideEvaluationArtifacts:
    """This class is used to set up any Snowflake server side artifacts for feedback evaluation."""

    def __init__(
        self,
        session: Session,
        database: str,
        schema: str,
        warehouse: str,
        database_prefix: str,
        use_staged_packages: bool,
    ) -> None:
        self._session = session
        self._database = database
        self._schema = schema
        self._warehouse = warehouse
        self._database_prefix = database_prefix
        self._use_staged_packages = use_staged_packages

    def set_up_all(self) -> None:
        if self._use_staged_packages:
            self._set_up_stage()
        self._set_up_stream()
        self._set_up_stored_procedure()
        self._set_up_wrapper_stored_procedure()
        self._set_up_task()

    def _run_query(self, q: str) -> None:
        cursor = self._session.connection.cursor()
        cursor.execute(q)
        cursor.fetchall()

    def _set_up_stage(self) -> None:
        self._run_query(f"CREATE STAGE IF NOT EXISTS {_STAGE_NAME}")
        data_directory = os.path.join(
            os.path.dirname(__file__), "../../../data/snowflake_stage_zips"
        )
        for trulens_package in (
            _TRULENS_PACKAGES + _TRULENS_EXTRA_STAGED_PACKAGES
        ):
            file_path = os.path.join(data_directory, f"{trulens_package}.zip")
            self._run_query(
                f"PUT file://{file_path} @{_STAGE_NAME} AUTO_COMPRESS = FALSE"
            )

    def _set_up_stream(self) -> None:
        self._run_query(
            f"""
            CREATE STREAM IF NOT EXISTS {_STREAM_NAME}
                ON TABLE {self._database}.{self._schema}.{self._database_prefix}FEEDBACKS
                SHOW_INITIAL_ROWS = TRUE
            """
        )

    def _set_up_stored_procedure(self) -> None:
        if self._use_staged_packages:
            import_packages = ",".join([
                f"'@{_STAGE_NAME}/{curr}.zip'" for curr in _TRULENS_PACKAGES
            ])
            import_statement = f"IMPORTS = ({import_packages})"
            packages_statement = ",".join([
                f"'{curr}'" for curr in _TRULENS_PACKAGES_DEPENDENCIES
            ])
        else:
            import_statement = ""
            packages_statement = ",".join([
                f"'{curr}'" for curr in _TRULENS_PACKAGES
            ])
        with open(_PYTHON_STORED_PROCEDURE_CODE_FILENAME, "r") as fh:
            python_code = fh.read()
        self._run_query(
            f"""
            CREATE PROCEDURE IF NOT EXISTS {_STORED_PROCEDURE_NAME}()
                RETURNS STRING
                LANGUAGE PYTHON
                RUNTIME_VERSION = '3.11'
                PACKAGES = (
                    {packages_statement}
                )
                {import_statement}
                HANDLER = 'run'
            AS
                $$\n{python_code}$$
            """
        )

    def _set_up_wrapper_stored_procedure(self) -> None:
        self._run_query(
            f"""
            CREATE PROCEDURE IF NOT EXISTS {_WRAPPER_STORED_PROCEDURE_NAME}()
                RETURNS VARCHAR
                LANGUAGE SQL
            AS
                $$
            BEGIN
                CALL {_STORED_PROCEDURE_NAME}();
                -- The following noop insert is done just so that the stream will clear.
                -- Currently, the only way for the stream to clear is for the data involved to be in a DML query.
                INSERT INTO {self._database_prefix}FEEDBACKS (
                    SELECT *
                    EXCLUDE (METADATA$ACTION, METADATA$ISUPDATE, METADATA$ROW_ID)
                    FROM {_STREAM_NAME}
                    WHERE FALSE
                );
            END;
                $$
            """
        )

    def _set_up_task(self) -> None:
        self._run_query(
            f"""
            CREATE TASK IF NOT EXISTS {_TASK_NAME}
                WAREHOUSE = {self._warehouse}
                SCHEDULE = '1 MINUTE'
                ALLOW_OVERLAPPING_EXECUTION = FALSE
                SUSPEND_TASK_AFTER_NUM_FAILURES = 0
                WHEN SYSTEM$STREAM_HAS_DATA('{_STREAM_NAME}')
                AS
                    CALL {_WRAPPER_STORED_PROCEDURE_NAME}()
            """
        )
        self._run_query(f"ALTER TASK {_TASK_NAME} RESUME")

import os
import re

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

_ZIPS_TO_UPLOAD = [
    "snowflake_sqlalchemy.zip",
    "trulens_connectors_snowflake.zip",
    "trulens_core.zip",
    "trulens_feedback.zip",
    "trulens_providers_cortex.zip",
]


class ServerSideEvaluationArtifacts:
    """This class is used to set up any Snowflake server side artifacts for feedback evaluation."""

    def __init__(
        self,
        session: Session,
        account: str,
        user: str,
        database: str,
        schema: str,
        warehouse: str,
        role: str,
        database_prefix: str,
    ) -> None:
        self._session = session
        self._account = account
        self._user = user
        self._database = database
        self._schema = schema
        self._warehouse = warehouse
        self._role = role
        self._database_prefix = database_prefix
        self._validate_name(database, "database")
        self._validate_name(schema, "schema")
        self._validate_name(warehouse, "warehouse")

    @staticmethod
    def _validate_name(name: str, error_message_variable_name: str) -> None:
        if not re.match(r"^[A-Za-z0-9_]+$", name):
            raise ValueError(
                f"`{error_message_variable_name}` must contain only alphanumeric and underscore characters!"
            )

    def set_up_all(self) -> None:
        self._set_up_stage()
        self._set_up_stream()
        self._set_up_stored_procedure()
        self._set_up_wrapper_stored_procedure()
        self._set_up_task()

    def _run_query(self, q: str) -> None:
        self._session.sql(q).collect()

    def _set_up_stage(self) -> None:
        self._run_query(f"CREATE STAGE IF NOT EXISTS {_STAGE_NAME}")
        data_directory = os.path.join(
            os.path.dirname(__file__), "../../../data/snowflake_stage_zips"
        )
        for zip_to_upload in _ZIPS_TO_UPLOAD:
            self._session.file.put(
                os.path.join(data_directory, zip_to_upload),
                f"@{_STAGE_NAME}",
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
        imports = ",".join([
            f"'@{_STAGE_NAME}/{curr}'" for curr in _ZIPS_TO_UPLOAD
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
                    -- TODO(dkurokawa): get these package versions automatically.
                    'alembic',
                    'dill',
                    'munch',
                    'nest-asyncio',
                    'nltk',
                    'numpy',
                    'packaging',
                    'pandas',
                    'pip',
                    'pydantic',
                    'python-dotenv',
                    'requests',
                    'rich',
                    'scikit-learn',
                    'scipy',
                    'snowflake-snowpark-python',
                    'sqlalchemy',
                    'tqdm',
                    'typing_extensions'
                )
                IMPORTS = ({imports})
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
                WHEN SYSTEM$STREAM_HAS_DATA('{_STREAM_NAME}')
                AS
                    CALL {_WRAPPER_STORED_PROCEDURE_NAME}()
            """
        )
        self._run_query(f"ALTER TASK {_TASK_NAME} RESUME")

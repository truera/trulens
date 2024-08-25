import os
import re

from snowflake.snowpark import Session

_ZIPS_TO_UPLOAD = [
    "snowflake_sqlalchemy.zip",
    "trulens_connectors_snowflake.zip",
    "trulens_core.zip",
    "trulens_feedback.zip",
    "trulens_providers_cortex.zip",
]


class ServerSideEvaluationArtifacts:
    def __init__(
        self,
        session: Session,
        account: str,
        user: str,
        database_name: str,
        schema_name: str,
        warehouse: str,
        role: str,
        database_url: str,
    ) -> None:
        self._session = session
        self._account = account
        self._user = user
        self._database_name = database_name
        self._schema_name = schema_name
        self._warehouse = warehouse
        self._role = role
        self._database_url = database_url
        self._validate_name(database_name, "database_name")
        self._validate_name(schema_name, "schema_name")
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

    @staticmethod
    def _remove_leading_spaces(q: str) -> str:
        # Get leading whitespace in first non-empty line.
        length_of_leading_whitespace = 0
        leading_whitespace_in_first_non_empty_line = ""
        for line in q.split("\n"):
            if line.strip():
                length_of_leading_whitespace = len(line) - len(line.lstrip())
                leading_whitespace_in_first_non_empty_line = line[
                    :length_of_leading_whitespace
                ]
                break
        # Remove the leading whitespace in the first non-empty line from all non-empty lines.
        ret = []
        for line in q.split("\n"):
            if line.startswith(leading_whitespace_in_first_non_empty_line):
                ret.append(line[length_of_leading_whitespace:])
            elif not line.strip():
                ret.append("")
            else:
                raise ValueError()
        return "\n".join(ret).strip()

    def _run_query(self, q: str) -> None:
        # Before running the query we first remove leading spaces for two reasons:
        # 1. Python code in a stored procedure needs to have correct indenting.
        # 2. To format the query for easier readability in debugging scenarios.
        self._session.sql(self._remove_leading_spaces(q)).collect()

    def _set_up_stage(self) -> None:
        self._run_query("CREATE STAGE IF NOT EXISTS TRULENS_PACKAGES_STAGE")
        data_directory = os.path.join(
            os.path.dirname(__file__), "../../../data/snowflake_stage_zips"
        )
        for zip_to_upload in _ZIPS_TO_UPLOAD:
            self._session.file.put(
                os.path.join(data_directory, zip_to_upload),
                "@TRULENS_PACKAGES_STAGE",
            )

    def _set_up_stream(self) -> None:
        self._run_query(
            f"""
            CREATE STREAM IF NOT EXISTS TRULENS_FEEDBACK_EVALS_STREAM
                ON TABLE {self._database_name}.{self._schema_name}.TRULENS_FEEDBACKS
                SHOW_INITIAL_ROWS = TRUE
            """
        )

    def _set_up_stored_procedure(self) -> None:
        stage_name = (
            f"{self._database_name}.{self._schema_name}.TRULENS_PACKAGES_STAGE"
        )
        imports = ",".join([
            f"'@{stage_name}/{curr}'" for curr in _ZIPS_TO_UPLOAD
        ])
        self._run_query(
            f"""
            CREATE PROCEDURE IF NOT EXISTS TRULENS_RUN_DEFERRED_FEEDBACKS()
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
                $$
            import _snowflake
            import trulens.providers.cortex.provider
            from snowflake.sqlalchemy import URL
            from trulens.core import TruSession
            from trulens.core.schema.feedback import FeedbackRunLocation


            def run(session):
                # Set up sqlalchemy engine parameters.
                conn = session.connection
                engine_params = {{}}
                engine_params["paramstyle"] = "qmark"
                engine_params["creator"] = lambda: conn
                database_args = {{"engine_params": engine_params}}
                # Ensure any Cortex provider uses the only Snowflake connection allowed in this stored procedure.
                trulens.providers.cortex.provider._SNOWFLAKE_STORED_PROCEDURE_CONNECTION = (
                    conn
                )
                # Run deferred feedback evaluator.
                db_url = URL(
                    account="{self._account}",
                    user="{self._user}",
                    password="password",
                    database="{self._database_name}",
                    schema="{self._schema_name}",
                    warehouse="{self._warehouse}",
                    role="{self._role}",
                )
                session = TruSession(
                    database_url=db_url,
                    database_check_revision=False,  # TODO: check revision in the future?
                    database_args=database_args,
                )
                session.start_evaluator(
                    run_location=FeedbackRunLocation.SNOWFLAKE,
                    return_when_done=True,
                    disable_tqdm=True,
                )
                $$
            """
        )

    def _set_up_wrapper_stored_procedure(self) -> None:
        self._run_query(
            """
            CREATE PROCEDURE IF NOT EXISTS TRULENS_RUN_DEFERRED_FEEDBACKS_WRAPPER()
                RETURNS VARCHAR
                LANGUAGE SQL
            AS
                $$
            BEGIN
                CALL TRULENS_RUN_DEFERRED_FEEDBACKS();
                -- The following noop insert is done just so that the stream will clear.
                -- Currently, the only way for the stream to clear is for the data involved to be in a DML query.
                INSERT INTO TRULENS_FEEDBACKS (
                    SELECT *
                    EXCLUDE (METADATA$ACTION, METADATA$ISUPDATE, METADATA$ROW_ID)
                    FROM TRULENS_FEEDBACK_EVALS_STREAM
                    WHERE FALSE
                );
            END;
                $$
            """
        )

    def _set_up_task(self) -> None:
        self._run_query(
            f"""
            CREATE TASK IF NOT EXISTS TRULENS_FEEDBACK_EVALS_TASK
                WAREHOUSE = {self._warehouse}
                SCHEDULE = '1 MINUTE'
                ALLOW_OVERLAPPING_EXECUTION = FALSE
                WHEN SYSTEM$STREAM_HAS_DATA('TRULENS_FEEDBACK_EVALS_STREAM')
                AS
                    CALL TRULENS_RUN_DEFERRED_FEEDBACKS_WRAPPER()
            """
        )
        self._run_query("ALTER TASK TRULENS_FEEDBACK_EVALS_TASK RESUME")

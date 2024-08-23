import os
import re

from snowflake.snowpark import Session

_ZIPS_TO_UPLOAD = [
    "snowflake_sqlalchemy.zip",
    "trulens_core.zip",
    "trulens_feedback.zip",
    "trulens_providers_cortex.zip",
]

_STORED_PROCEDURE_PYTHON_CODE = """
import _snowflake
import trulens.providers.cortex.provider
from trulens.core import Tru
from trulens.core.schema.feedback import FeedbackRunLocation

def run(session):
    # Set up sqlalchemy engine parameters.
    conn = session._conn._conn # TODO(this_pr): Can't I just say session.connection?
    engine_params = {}
    engine_params["paramstyle"] = "qmark"
    engine_params["creator"] = lambda: conn
    # Ensure any Cortex provider uses the only Snowflake connection allowed in this stored procedure.
    trulens.providers.cortex.provider._SNOWFLAKE_STORED_PROCEDURE_CONNECTION = conn
    # Run deferred feedback evaluator.
    db_url = _snowflake.get_generic_secret_string("trulens_db_url")
    tru = Tru(database_url=db_url, database_check_revision=False, sqlalchemy_engine_params=engine_params)  # TODO(this_pr): Remove database_check_revision.
    tru.start_evaluator(run_location=FeedbackRunLocation.SNOWFLAKE, return_when_done=True, disable_tqdm=True)
"""


class SnowflakeServerSideEvaluationArtifacts:
    def __init__(
        self,
        session: Session,
        database_name: str,
        schema_name: str,
        warehouse_name: str,
        external_access_integration_name: str,
        database_url: str,
    ):
        self._session = session
        self._database_name = database_name
        self._schema_name = schema_name
        self._warehouse_name = warehouse_name
        self._external_access_integration_name = (
            external_access_integration_name
        )
        self._database_url = database_url
        self._validate_name(database_name, "database_name")
        self._validate_name(schema_name, "schema_name")
        self._validate_name(warehouse_name, "warehouse_name")
        self._validate_name(
            external_access_integration_name, "external_access_integration_name"
        )

    @staticmethod
    def _validate_name(name: str, error_message_variable_name: str):
        if not re.match(r"^[A-Za-z0-9_]+$", name):
            raise ValueError(
                f"`{error_message_variable_name}` must contain only alphanumeric and underscore characters!"
            )

    def set_up_all(self):
        self._set_up_stage()
        self._set_up_stream()
        self._set_up_secret()
        self._set_up_external_access_integration()
        self._set_up_stored_procedure()
        self._set_up_task()

    def _set_up_stage(self):
        self._session.sql(
            "CREATE STAGE IF NOT EXISTS TRULENS_PACKAGES_STAGE"
        ).collect()
        data_directory = os.path.join(
            os.path.dirname(__file__), "../../data/snowflake_stage_zips"
        )
        for zip_to_upload in _ZIPS_TO_UPLOAD:
            self._session.file.put(
                os.path.join(data_directory, zip_to_upload),
                "@TRULENS_PACKAGES_STAGE",
            )

    def _set_up_stream(self):
        self._session.sql(
            f"""
            CREATE STREAM IF NOT EXISTS TRULENS_FEEDBACK_EVALS_STREAM
                ON TABLE {self._database_name}.{self._schema_name}.TRULENS_FEEDBACKS
            """
        ).collect()

    def _set_up_secret(self):
        self._session.sql(
            """
            CREATE SECRET IF NOT EXISTS TRULENS_DB_URL
                TYPE = GENERIC_STRING
                SECRET_STRING = ?
            """,
            params=[self._database_url],
        ).collect()

    def _set_up_external_access_integration(self):
        self._session.sql(
            """
            CREATE NETWORK RULE IF NOT EXISTS TRULENS_DUMMY_NETWORK_RULE
                TYPE = HOST_PORT
                MODE = EGRESS
                VALUE_LIST = ('snowflake.com')
                COMMENT = 'This is a dummy network rule created entirely because secrets cannot be used without one.'
            """
        ).collect()
        self._session.sql(
            f"""
            CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS {self._external_access_integration_name}
                ALLOWED_NETWORK_RULES = (TRULENS_DUMMY_NETWORK_RULE)
                ALLOWED_AUTHENTICATION_SECRETS = (TRULENS_DB_URL)
                ENABLED = TRUE
                COMMENT = 'This is a dummy EAI created entirely because secrets cannot be used without one.'
            """
        ).collect()

    def _set_up_stored_procedure(self):
        stage_name = (
            f"{self._database_name}.{self._schema_name}.TRULENS_PACKAGES_STAGE"
        )
        imports = ",".join([
            f"'@{stage_name}/{curr}'" for curr in _ZIPS_TO_UPLOAD
        ])
        self._session.sql(
            f"""
            CREATE PROCEDURE IF NOT EXISTS TRULENS_RUN_DEFERRED_FEEDBACKS()
                RETURNS STRING
                LANGUAGE PYTHON
                RUNTIME_VERSION = '3.11'
                PACKAGES = (
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
                SECRETS = ('trulens_db_url' = TRULENS_DB_URL)
                EXTERNAL_ACCESS_INTEGRATIONS = ({self._external_access_integration_name})
            AS
                $$
                {_STORED_PROCEDURE_PYTHON_CODE}
                $$
            """
        ).collect()

    def _set_up_task(self):
        self._session.sql(
            f"""
            CREATE TASK IF NOT EXISTS TRULENS_FEEDBACK_EVAL_TASK
                WAREHOUSE = {self._warehouse_name}
                SCHEDULE = '1 MINUTE'
                ALLOW_OVERLAPPING_EXECUTION = FALSE
                WHEN SYSTEM$STREAM_HAS_DATA('TRULENS_FEEDBACK_EVALS_STREAM')
                AS
                    CALL TRULENS_RUN_DEFERRED_FEEDBACKS()
            """
        ).collect()
        self._session.sql(
            "ALTER TASK TRULENS_FEEDBACK_EVAL_TASK RESUME"
        ).collect()


# TODO(this_pr): Ensure we use the right trulens version.

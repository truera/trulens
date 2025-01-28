import os
import time
from typing import Dict, Sequence
import unittest
import uuid

from snowflake.snowpark import Session
from trulens.apps.llamaindex import TruLlama
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.experimental import Feature
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.exporter.snowflake import (
    TruLensSnowflakeSpanExporter,
)

from tests.unit.test_otel_tru_llama import TestOtelTruLlama


class TestSnowflakeEventTableExporter(unittest.TestCase):
    # TODO: this is duplicated.
    @classmethod
    def clear_TruSession_singleton(cls) -> None:
        # [HACK!] Clean up any instances of `TruSession` so tests don't
        # interfere with each other.
        for key in [
            curr
            for curr in TruSession._singleton_instances
            if curr[0] == "trulens.core.session.TruSession"
        ]:
            del TruSession._singleton_instances[key]

    @staticmethod
    def _create_snowpark_session():
        _snowflake_connection_parameters: Dict[str, str] = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ["SNOWFLAKE_DATABASE"],
            "schema": os.environ["SNOWFLAKE_SCHEMA"],
            "role": os.environ["SNOWFLAKE_ROLE"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        }
        return Session.builder.configs(
            _snowflake_connection_parameters
        ).create()

    @staticmethod
    def _create_db_connector(snowpark_session: Session):
        return snowflake_connector.SnowflakeConnector(
            snowpark_session=snowpark_session,
            init_server_side=False,
            init_server_side_with_staged_packages=False,
        )

    # TODO: this is duplicated, mostly.
    @classmethod
    def setUpClass(cls) -> None:
        # Set up Snowflake.
        cls.snowpark_session = cls._create_snowpark_session()
        cls.db_connector = cls._create_db_connector(cls.snowpark_session)
        # Set up OTEL.
        cls.exporter = TruLensSnowflakeSpanExporter(cls.db_connector)
        # Set up TruSession.
        os.environ["TRULENS_OTEL_TRACING"] = "1"
        cls.clear_TruSession_singleton()
        cls.tru_session = TruSession(
            cls.db_connector,
            experimental_feature_flags=[Feature.OTEL_TRACING],
            _experimental_otel_exporter=cls.exporter,
        )
        cls.tru_session.experimental_enable_feature("otel_tracing")
        return super().setUpClass()

    # TODO: this is duplicated.
    @classmethod
    def tearDownClass(cls) -> None:
        cls.clear_TruSession_singleton()
        cls.snowpark_session.close()
        return super().tearDownClass()

    @classmethod
    def _run_query(cls, q: str):
        return cls.snowpark_session.sql(q).collect()

    @classmethod
    def _wait_for_num_results(
        cls,
        q: str,
        expected_num_results: int,
        num_retries: int = 30,
        retry_cooldown_in_seconds: int = 10,
    ) -> Sequence:
        for _ in range(num_retries):
            results = cls._run_query(q)
            if len(results) == expected_num_results:
                return results
            time.sleep(retry_cooldown_in_seconds)
        raise ValueError("Did not get the expected number of results!")

    def test_llama_index(self):
        # Create app.
        rag = TestOtelTruLlama._create_simple_rag()
        tru_recorder = TruLlama(
            rag,
            app_name="Simple RAG",
            app_version="v1",
        )
        # Record and invoke.
        run_name = str(uuid.uuid4())
        with tru_recorder(run_name=run_name, input_id="42"):
            rag.query("What is multi-headed attention?")
        # Flush exporter and wait for data to be made to stage.
        self.tru_session.experimental_force_flush()
        # Check that the data is in the event table.
        self._wait_for_num_results(
            f"""
            SELECT
                *
            FROM EVENT_DB.PUBLIC.EVENTS
            WHERE
                RECORD_TYPE = 'SPAN'
                AND TIMESTAMP >= TO_TIMESTAMP_LTZ('2025-01-21 23:00:00')
                -- AND TIMESTAMP <=TO_TIMESTAMP_LTZ('2025-01-11 23:00:00')
                AND RECORD_ATTRIBUTES['ai_observability.run_name'] = '{run_name}'
            ORDER BY TIMESTAMP DESC
            LIMIT 50
        """,
            13,  # TODO: get this from the exporter or something?
        )
        # TODO: call the feedback computation and check that it's fine.

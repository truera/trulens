import logging
import os
import time
from typing import Sequence
import uuid

from snowflake.snowpark import Session
from trulens.apps.llamaindex import TruLlama
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import BASE_SCOPE

from tests.unit.test_otel_tru_llama import TestOtelTruLlama
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestSnowflakeEventTableExporter(SnowflakeTestCase):
    logger = logging.getLogger(__name__)

    @staticmethod
    def _create_db_connector(snowpark_session: Session):
        return snowflake_connector.SnowflakeConnector(
            snowpark_session=snowpark_session,
        )

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["TRULENS_OTEL_TRACING"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["TRULENS_OTEL_TRACING"]
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.create_and_use_schema(
            "TestSnowflakeEventTableExporter", append_uuid=True
        )
        db_connector = self._create_db_connector(self._snowpark_session)
        self._tru_session = TruSession(db_connector)

    def _wait_for_num_results(
        self,
        q: str,
        expected_num_results: int,
        num_retries: int = 30,
        retry_cooldown_in_seconds: int = 10,
    ) -> Sequence:
        for _ in range(num_retries):
            results = self.run_query(q)
            self.logger.info(
                f"Got {len(results)} results, expecting {expected_num_results}"
            )
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
        self._tru_session.experimental_force_flush()
        # Check that there are no other tables in the schema.
        self.assertListEqual(
            self.run_query("SHOW TABLES"),
            [],
        )
        # Check that the data is in the event table.
        self._wait_for_num_results(
            f"""
            SELECT
                *
            FROM EVENT_DB.PUBLIC.EVENTS
            WHERE
                RECORD_TYPE = 'SPAN'
                AND TIMESTAMP >= TO_TIMESTAMP_LTZ('2025-01-28 00:00:00')
                AND RECORD_ATTRIBUTES['{BASE_SCOPE}.run_name'] = '{run_name}'
            ORDER BY TIMESTAMP DESC
            LIMIT 50
        """,
            8,  # TODO(otel): get this from the exporter or something?
        )
        # TODO(otel): call the feedback computation and check that it's fine.

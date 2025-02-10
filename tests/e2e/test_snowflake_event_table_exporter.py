import json
import logging
import os
import time
from typing import List, Sequence
import uuid

from snowflake.snowpark import Session
from snowflake.snowpark.row import Row
from trulens.apps.custom import TruCustomApp
from trulens.apps.langchain import TruChain
from trulens.apps.llamaindex import TruLlama
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.session import TruSession
from trulens.feedback.computer import MinimalSpanInfo
from trulens.feedback.computer import RecordGraphNode
from trulens.feedback.computer import _compute_feedback
from trulens.otel.semconv.trace import SpanAttributes

import tests.unit.test_otel_tru_chain
import tests.unit.test_otel_tru_custom
import tests.unit.test_otel_tru_llama
from tests.util.mock_otel_feedback_computation import (
    all_retrieval_span_attributes,
)
from tests.util.mock_otel_feedback_computation import feedback_function
from tests.util.snowflake_test_case import SnowflakeTestCase


def _convert_events_to_MinimalSpanInfos(
    events: List[Row],
) -> List[MinimalSpanInfo]:
    ret = []
    for row in events:
        span = MinimalSpanInfo()
        span.span_id = json.loads(row.TRACE)["span_id"]
        span.parent_span_id = json.loads(row.RECORD).get("parent_span_id", None)
        if not span.parent_span_id:
            span.parent_span_id = None
        span.attributes = json.loads(row.RECORD_ATTRIBUTES)
        ret.append(span)
    return ret


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
        params: List[str],
        expected_num_results: int,
        num_retries: int = 15,
        retry_cooldown_in_seconds: int = 10,
    ) -> Sequence:
        for _ in range(num_retries):
            results = self.run_query(q, params)
            if len(results) == expected_num_results:
                return results
            self.logger.info(
                f"Got {len(results)} results, expecting {expected_num_results}"
            )
            time.sleep(retry_cooldown_in_seconds)
        raise ValueError(
            f"Did not get the expected number of results! Expected {expected_num_results} results, but last found: {len(results)}! The results:\n{results}"
        )

    def _validate_results(
        self, app_name: str, run_name: str, num_expected_spans: int
    ):
        # Flush exporter and wait for data to be made to stage.
        self._tru_session.force_flush()
        # Check that there are no other tables in the schema.
        self.assertListEqual(
            self.run_query("SHOW TABLES"),
            [],
        )
        # Check that the data is in the event table.
        return self._wait_for_num_results(
            f"""
            SELECT
                *
            FROM
                table(snowflake.local.GET_AI_OBSERVABILITY_EVENTS(
                    ?,
                    ?,
                    ?,
                    'EXTERNAL AGENT'
                ))
            WHERE
                RECORD_TYPE = 'SPAN'
                AND TIMESTAMP >= TO_TIMESTAMP_LTZ('2025-01-31 20:42:00')
                AND RECORD_ATTRIBUTES['{SpanAttributes.RUN_NAME}'] = '{run_name}'
            ORDER BY TIMESTAMP DESC
            LIMIT 50
            """,
            [
                self._snowpark_session.get_current_database()[1:-1],
                self._snowpark_session.get_current_schema()[1:-1],
                app_name,
            ],
            num_expected_spans,
        )
        # TODO(otel): call the feedback computation and check that it's fine.

    def test_tru_custom_app(self):
        # Create app.
        app = tests.unit.test_otel_tru_custom.TestApp()
        tru_recorder = TruCustomApp(
            app,
            app_name="custom app",
            app_version="v1",
            main_method=app.respond_to_query,
        )
        # Record and invoke.
        run_name = str(uuid.uuid4())
        with tru_recorder(run_name=run_name, input_id="42"):
            app.respond_to_query("Kojikun")
        # Record and invoke again.
        self._tru_session.force_flush()
        with tru_recorder(run_name=run_name, input_id="21"):
            app.respond_to_query("Nolan")
        # Validate results.
        self._validate_results("custom app", run_name, 10)

    def test_tru_llama(self):
        # Create app.
        rag = (
            tests.unit.test_otel_tru_llama.TestOtelTruLlama._create_simple_rag()
        )
        tru_recorder = TruLlama(
            rag,
            app_name="llama-index app",
            app_version="v1",
            main_method=rag.query,
        )
        # Record and invoke.
        run_name = str(uuid.uuid4())
        with tru_recorder(run_name=run_name, input_id="42"):
            rag.query("What is multi-headed attention?")
        # Validate results.
        self._validate_results("llama-index app", run_name, 8)

    def test_tru_chain(self):
        # Create app.
        rag = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        tru_recorder = TruChain(
            rag,
            app_name="langchain app",
            app_version="v1",
            main_method=rag.invoke,
        )
        # Record and invoke.
        run_name = str(uuid.uuid4())
        with tru_recorder(run_name=run_name, input_id="42"):
            rag.invoke("What is multi-headed attention?")
        # Validate results.
        self._validate_results("langchain app", run_name, 10)

    def test_feedback_computation(self) -> None:
        # Create app.
        rag_chain = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        app_name = "Simple RAG"
        tru_recorder = TruChain(
            rag_chain,
            app_name=app_name,
            app_version="v1",
            main_method=rag_chain.invoke,
        )
        # Record and invoke.
        run_name = str(uuid.uuid4())
        with tru_recorder(run_name=run_name, input_id="42"):
            rag_chain.invoke("What is multi-headed attention?")
        TruSession().force_flush()
        # Compute feedback on record we just ingested.
        events = self._validate_results(app_name, run_name, 10)
        spans = _convert_events_to_MinimalSpanInfos(events)
        record_root = RecordGraphNode.build_graph(spans)
        _compute_feedback(
            record_root, feedback_function, all_retrieval_span_attributes
        )
        TruSession().force_flush()
        # Validate results.
        events = self._validate_results(app_name, run_name, 13)

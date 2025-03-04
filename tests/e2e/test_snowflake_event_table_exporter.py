import logging
import os
import time
from typing import Any, Callable, Dict, List, Tuple, Type
import uuid

import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.row import Row
from trulens.apps.app import TruApp
from trulens.apps.langchain import TruChain
from trulens.apps.llamaindex import TruLlama
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.app import App
from trulens.core.run import Run
from trulens.core.run import RunConfig
from trulens.core.session import TruSession

import tests.unit.test_otel_tru_chain
import tests.unit.test_otel_tru_custom
import tests.unit.test_otel_tru_llama
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
        params: List[str],
        expected_num_results: int,
        num_retries: int = 15,
        retry_cooldown_in_seconds: int = 10,
    ) -> List[Row]:
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
        self, app_name: str, num_expected_spans: int
    ) -> List[Row]:
        # Flush exporter and wait for data to be made to stage.
        self._tru_session.force_flush()
        # Check that there are no other tables in the schema.
        self.assertListEqual(self.run_query("SHOW TABLES"), [])
        # Check that the data is in the event table.
        return self._wait_for_num_results(
            """
            SELECT
                *
            FROM
                table(snowflake.local.GET_AI_OBSERVABILITY_EVENTS(
                    ?,
                    ?,
                    ?,
                    'EXTERNAL AGENT'
                ))
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

    def _test_tru_app(
        self,
        app: Any,
        main_method: Callable,
        TruAppClass: Type[App],
        dataset_spec: Dict[str, str],
        input_df: pd.DataFrame,
        num_expected_spans: int,
    ) -> Tuple[str, Run]:
        # Create app.
        app_name = str(uuid.uuid4())
        app_name = app_name.upper()  # TODO(this_pr): remove!
        tru_recorder = TruAppClass(
            app,
            app_name=app_name,
            app_version="v1",
            connector=self._tru_session.connector,
            main_method=main_method,
        )
        # Create run.
        run_name = str(uuid.uuid4())
        run_config = RunConfig(
            run_name=run_name,
            description="desc",
            dataset_name="My test dataframe name",
            source_type="DATAFRAME",
            label="label",
            dataset_spec=dataset_spec,
        )
        # Record and invoke.
        run = tru_recorder.add_run(run_config=run_config)
        run.start(input_df=input_df)
        self._tru_session.force_flush()
        # Validate results.
        self._validate_results(app_name, num_expected_spans)
        return app_name, run

    def test_tru_custom_app(self):
        app = tests.unit.test_otel_tru_custom.TestApp()
        self._test_tru_app(
            app,
            app.respond_to_query,
            TruApp,
            {"input": "custom_input"},
            pd.DataFrame({"custom_input": ["Kojikun", "Nolan"]}),
            8,
        )

    def test_tru_llama(self):
        app = (
            tests.unit.test_otel_tru_llama.TestOtelTruLlama._create_simple_rag()
        )
        self._test_tru_app(
            app,
            app.query,
            TruLlama,
            {"input": "custom_input"},
            pd.DataFrame({"custom_input": ["What is multi-headed attention?"]}),
            7,
        )

    def test_tru_chain(self):
        app = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        self._test_tru_app(
            app,
            app.invoke,
            TruChain,
            {"input": "custom_input"},
            pd.DataFrame({"custom_input": ["What is multi-headed attention?"]}),
            9,
        )

    def test_feedback_computation(self) -> None:
        app = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        app_name, run = self._test_tru_app(
            app,
            app.invoke,
            TruChain,
            {
                "input": "custom_input",
                "ground_truth_output": "expected_response",
            },
            pd.DataFrame({
                "custom_input": ["What is multi-headed attention?"],
                "expected_response": ["Like attention but with more heads."],
            }),
            9,
        )
        run.compute_metrics([
            "context_relevance",
            "groundedness",
            "answer_relevance",
            "coherence",
            # "correctness",
        ])
        self._validate_results(app_name, 20)

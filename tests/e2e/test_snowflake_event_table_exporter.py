import logging
import os
from typing import Any, Callable, Dict, Tuple, Type
import uuid

import pandas as pd
from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.apps.langchain import TruChain
from trulens.apps.llamaindex import TruLlama
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.app import App
from trulens.core.otel.instrument import instrument
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
        instrument.enable_all_instrumentation()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        instrument.disable_all_instrumentation()
        del os.environ["TRULENS_OTEL_TRACING"]
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.create_and_use_schema(
            "TestSnowflakeEventTableExporter", append_uuid=True
        )
        db_connector = self._create_db_connector(self._snowpark_session)
        self._tru_session = TruSession(db_connector)

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
        self._validate_num_spans_for_app(app_name, num_expected_spans)
        return app_name, run

    def test_tru_custom_app(self):
        app = tests.unit.test_otel_tru_custom.TestApp()
        self._test_tru_app(
            app,
            app.respond_to_query,
            TruApp,
            {"input": "custom_input"},
            pd.DataFrame({"custom_input": ["test", "throw"]}),
            pd.read_csv(
                "tests/unit/static/golden/test_otel_tru_custom__test_smoke.csv",
                index_col=0,
            ).shape[0],
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
            pd.read_csv(
                "tests/unit/static/golden/test_otel_tru_llama__test_smoke.csv",
                index_col=0,
            ).shape[0],
        )

    def test_tru_chain(self):
        app = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        self._test_tru_app(
            app,
            app.invoke,
            TruChain,
            {"record_root.input": "custom_input"},
            pd.DataFrame({"custom_input": ["What is multi-headed attention?"]}),
            pd.read_csv(
                "tests/unit/static/golden/test_otel_tru_chain__test_smoke.csv",
                index_col=0,
            ).shape[0],
        )

    def test_feedback_computation(self) -> None:
        app = (
            tests.unit.test_otel_tru_chain.TestOtelTruChain._create_simple_rag()
        )
        input_df = pd.DataFrame({
            "custom_input": ["What is multi-headed attention?"],
            "expected_response": ["Like attention but with more heads."],
        })
        app_name, run = self._test_tru_app(
            app,
            app.invoke,
            TruChain,
            {
                "input": "custom_input",
                "ground_truth_output": "expected_response",
            },
            input_df,
            pd.read_csv(
                "tests/unit/static/golden/test_otel_tru_chain__test_smoke.csv",
                index_col=0,
            ).shape[0],
        )
        feedbacks_to_compute = [
            "context_relevance",
            "groundedness",
            "answer_relevance",
            "coherence",
            "correctness",
        ]
        run.compute_metrics(feedbacks_to_compute)
        self._validate_num_spans_for_app(app_name, 23)
        records, feedbacks = self._tru_session.get_records_and_feedback(
            app_name=app_name, app_version="v1"
        )
        self.assertEqual(sorted(feedbacks_to_compute), sorted(feedbacks))
        self.assertEqual(input_df.shape[0], records.shape[0])
        self.assertEqual(
            feedbacks_to_compute,
            [
                curr
                for curr in records.columns.tolist()
                if curr in feedbacks_to_compute
            ],
        )

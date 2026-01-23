import json
import logging
import os
from typing import Any, Callable, Dict, Tuple, Type
import uuid

import pandas as pd
from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.apps.langchain import TruChain
from trulens.apps.langgraph import TruGraph
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

    def test_tru_graph_with_json_state_input(self):
        """
        Test TruGraph with JSON state blobs as input (simulating Snowflake VARIANT columns).

        This verifies that the Run API correctly parses JSON strings into dict state objects
        and passes them to TruGraph's main_call method.
        """
        from langchain_core.messages import AIMessage
        from langgraph.graph import MessagesState
        from langgraph.graph import StateGraph

        # Create a simple LangGraph app
        def echo_node(state: MessagesState):
            last_message = state["messages"][-1]
            if hasattr(last_message, "content"):
                content = last_message.content
            elif (
                isinstance(last_message, (list, tuple))
                and len(last_message) > 1
            ):
                content = last_message[1]
            else:
                content = str(last_message)
            return {"messages": [AIMessage(content=f"Echo: {content}")]}

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("echo", echo_node)
        graph_builder.set_entry_point("echo")
        graph_builder.set_finish_point("echo")
        graph = graph_builder.compile()

        # Create TruGraph wrapper
        app_name = str(uuid.uuid4())
        tru_recorder = TruGraph(
            graph,
            app_name=app_name,
            app_version="v1",
            connector=self._tru_session.connector,
        )

        # Create DataFrame with JSON state blobs (simulating Snowflake VARIANT column)
        # The JSON strings represent LangGraph state dicts
        input_df = pd.DataFrame({
            "state_input": [
                json.dumps({"messages": [["user", "Hello world"]]}),
                json.dumps({"messages": [["user", "Test message"]]}),
            ]
        })

        # Create RunConfig mapping to the JSON column
        run_name = str(uuid.uuid4())
        run_config = RunConfig(
            run_name=run_name,
            description="Test TruGraph with JSON state inputs",
            dataset_name="test_langgraph_states",
            source_type="DATAFRAME",
            label="langgraph_test",
            dataset_spec={"input": "state_input"},
        )

        # Start the run - this should:
        # 1. Parse JSON strings into dict state objects (in run.py)
        # 2. Pass dict to TruGraph's main_call (which accepts Union[str, dict])
        # 3. TruGraph invokes graph with the state dict
        run = tru_recorder.add_run(run_config=run_config)
        run.start(input_df=input_df)
        self._tru_session.force_flush()

        # Validate that spans were created (at least 2 root spans for 2 inputs)
        # The exact number depends on LangGraph's internal span creation
        self._validate_num_spans_for_app(app_name, num_expected_spans=2)

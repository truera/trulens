from datetime import datetime
import json
import logging
import math
import os
import time
from typing import List, Optional

import numpy as np
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace.status import Status
from opentelemetry.trace.status import StatusCode
import pandas as pd
from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.connectors import snowflake as snowflake_connector
from trulens.core.otel.instrument import instrument
from trulens.core.run import Run
from trulens.core.run import RunConfig
from trulens.core.run import RunStatus
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.snowflake_test_case import SnowflakeTestCase

logger = logging.getLogger(__name__)


class LoadTestApp:
    def __init__(self, data_filename: str) -> None:
        self._data = pd.read_csv(data_filename)
        self._data["context"] = self._data["context"].apply(json.loads)
        for context in self._data["context"]:
            if list(context.keys()) != ["sentences"]:
                raise ValueError()
            if len(context["sentences"]) != 3:
                raise ValueError()
            for sentence in context["sentences"]:
                if not isinstance(sentence, str):
                    raise ValueError()
        self._data["context"] = self._data["context"].apply(
            lambda curr: curr["sentences"]
        )
        self._current_row = None

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def get_context(self, query: str) -> List[str]:
        for _, row in self._data.iterrows():
            if query == row["query"]:
                self._current_row = row
                return row["context"]
        raise ValueError(f"Could not find query: {query}")

    @instrument()
    def query_with_context(self, query: str, context: List[str]) -> str:
        prompt = "Please answer the following query given the context:\n\n"
        prompt += f"QUERY\n:{query}\n\n"
        prompt += f"CONTEXT\n:{context}\n\n"
        return self.generate(prompt)

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, prompt: str) -> str:
        return self._current_row["response"]

    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def query(self, query: str) -> str:
        return self.query_with_context(query, self.get_context(query))


class TestSnowflake(SnowflakeTestCase):
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
            "TestSnowflake", append_uuid=True, delete_schema_on_cleanup=False
        )
        db_connector = self._create_db_connector(self._snowpark_session)
        self._tru_session = TruSession(db_connector)

    @staticmethod
    def _wait_for_run_to_finish(
        run: Run, timeout_in_seconds: float = 120
    ) -> None:
        start_time = datetime.now()
        while (
            datetime.now() - start_time
        ).total_seconds() < timeout_in_seconds:
            if run.get_status() in [
                RunStatus.COMPLETED,
                RunStatus.INVOCATION_COMPLETED,
            ]:
                return
            if run.get_status() == RunStatus.FAILED:
                raise ValueError("Run FAILED!")
            time.sleep(1)
        raise ValueError("Run did not finish in time!")

    def _wait_for_events(
        self,
        num_expected_events: int,
        app_name: str,
        num_retries: int = 10,
        retry_cooldown_in_seconds: int = 5,
        return_events: bool = False,
    ) -> Optional[pd.DataFrame]:
        q = """
            SELECT
                COUNT(*) AS NUM_EVENTS
            FROM
                table(snowflake.local.GET_AI_OBSERVABILITY_EVENTS(
                    ?, ?, ?, 'EXTERNAL AGENT'
                ))
            """
        params = [
            self._snowpark_session.get_current_database()[1:-1],
            self._snowpark_session.get_current_schema()[1:-1],
            app_name,
        ]
        for _ in range(num_retries):
            res = self._snowpark_session.sql(q, params=params).to_pandas()
            num_events = res.iloc[0].NUM_EVENTS
            if num_events == num_expected_events:
                if not return_events:
                    return None
                q = q.replace("COUNT(*) AS NUM_EVENTS", "*")
                ret = self._snowpark_session.sql(q, params=params).to_pandas()
                for json_col in [
                    "TRACE",
                    "RESOURCE_ATTRIBUTES",
                    "RECORD",
                    "RECORD_ATTRIBUTES",
                ]:
                    ret[json_col] = ret[json_col].apply(json.loads)
                if len(ret) != num_expected_events:
                    raise ValueError("Unexpected number of events!")
                return ret
            time.sleep(retry_cooldown_in_seconds)
        raise ValueError(
            f"Found {num_events} events, but expected {num_expected_events}!"
        )

    @staticmethod
    def _sort_events_by_record_id(events: pd.DataFrame) -> None:
        events.sort_values(
            by=["RECORD_ATTRIBUTES", "START_TIMESTAMP"],
            key=lambda ser: (
                ser.apply(lambda curr: curr[SpanAttributes.RECORD_ID])
                if isinstance(ser.iloc[0], dict)
                else ser
            ),
            inplace=True,
        )

    @staticmethod
    def _random_id64() -> str:
        return str(np.random.randint(0, 2**63))

    @staticmethod
    def _update_events(
        events: pd.DataFrame, app_name: str, run_name: str
    ) -> None:
        parent_span_id = ""
        for _, curr in events.iterrows():
            span_id = TestSnowflake._random_id64()
            if curr["RECORD_ATTRIBUTES"][SpanAttributes.SPAN_TYPE] in [
                SpanAttributes.SpanType.RECORD_ROOT,
                SpanAttributes.SpanType.EVAL_ROOT,
            ]:
                parent_span_id = ""
                trace_id = TestSnowflake._random_id64()
            if (
                curr["RECORD_ATTRIBUTES"][SpanAttributes.SPAN_TYPE]
                == SpanAttributes.SpanType.RECORD_ROOT
            ):
                record_id = TestSnowflake._random_id64()
            if (
                curr["RECORD_ATTRIBUTES"][SpanAttributes.SPAN_TYPE]
                == SpanAttributes.SpanType.EVAL_ROOT
            ):
                eval_root_id = span_id
            curr["TRACE"]["trace_id"] = trace_id
            curr["TRACE"]["span_id"] = span_id
            if "parent_span_id" in curr["RECORD"]:
                if not parent_span_id:
                    raise ValueError()
                curr["RECORD"]["parent_span_id"] = parent_span_id
            curr["RECORD_ATTRIBUTES"][SpanAttributes.APP_NAME] = app_name
            curr["RECORD_ATTRIBUTES"][SpanAttributes.RUN_NAME] = run_name
            curr["RECORD_ATTRIBUTES"][SpanAttributes.RECORD_ID] = record_id
            if (
                SpanAttributes.EVAL.TARGET_RECORD_ID
                in curr["RECORD_ATTRIBUTES"]
            ):
                curr["RECORD_ATTRIBUTES"][
                    SpanAttributes.EVAL.TARGET_RECORD_ID
                ] = record_id
                curr["RECORD_ATTRIBUTES"][SpanAttributes.EVAL.EVAL_ROOT_ID] = (
                    eval_root_id
                )
            parent_span_id = span_id

    def _ingest_app_and_run(
        self,
        app: LoadTestApp,
        app_name: str,
        run_config: RunConfig,
        run_name: str,
    ) -> None:
        tru_app = TruApp(
            app,
            app_name=app_name,
            app_version="V1",
            connector=self._tru_session.connector,
        )
        run_config.run_name = run_name
        tru_app.add_run(run_config=run_config)
        # TODO(this_pr): should we be doing something regarding the run state?

    def _ingest_events(self, events: pd.DataFrame) -> None:
        spans = []
        for _, event in events.iterrows():
            # Compute context.
            context = trace.SpanContext(
                trace_id=int(event["TRACE"]["trace_id"]),
                span_id=int(event["TRACE"]["span_id"]),
                is_remote=False,
            )
            # Compute parent.
            parent = None
            if "parent_span_id" in event["RECORD"]:
                parent = trace.SpanContext(
                    trace_id=int(event["TRACE"]["trace_id"]),
                    span_id=int(event["RECORD"]["parent_span_id"]),
                    is_remote=False,
                )
            # Compute kind.
            kind = trace.SpanKind.INTERNAL
            if "kind" in event["RECORD"]:
                {
                    "SPAN_KIND_UNSPECIFIED": trace.SpanKind.INTERNAL,
                    "SPAN_KIND_INTERNAL": trace.SpanKind.INTERNAL,
                    "SPAN_KIND_CLIENT": trace.SpanKind.CLIENT,
                    "SPAN_KIND_CONSUMER": trace.SpanKind.CONSUMER,
                    "SPAN_KIND_PRODUCER": trace.SpanKind.PRODUCER,
                    "SPAN_KIND_SERVER": trace.SpanKind.SERVER,
                }[event["RECORD"]["kind"]]
            # Compute status.
            status = Status(StatusCode.UNSET)
            if (
                "status" in event["RECORD"]
                and "code" in event["RECORD"]["status"]
            ):
                status = Status(
                    {
                        "STATUS_CODE_UNSET": StatusCode.UNSET,
                        "STATUS_CODE_OK": StatusCode.OK,
                        "STATUS_CODE_ERROR": StatusCode.ERROR,
                    }[event["RECORD"]["status"]["code"]]
                )
            # Handle "snow.*" attributes.
            attributes = event["RECORD_ATTRIBUTES"]
            attributes[SpanAttributes.APP_VERSION] = attributes[
                "snow.ai.observability.object.version.name"
            ]
            attributes = {
                k: v for k, v in attributes.items() if not k.startswith("snow.")
            }
            # Create `ReadableSpan`.
            span = ReadableSpan(
                name=event["RECORD"]["name"],
                context=context,
                parent=parent,
                resource=Resource(event["RESOURCE_ATTRIBUTES"]),
                attributes=attributes,
                kind=kind,
                status=status,
                start_time=int(event["START_TIMESTAMP"].timestamp() * 1e9),
                end_time=int(event["TIMESTAMP"].timestamp() * 1e9),
            )
            spans.append(span)
        self._tru_session.experimental_otel_exporter.export(spans)

    def _test_ingest_data(
        self,
        data_filename: str,
        num_apps: int,
        num_runs: int,
        num_inputs: int,
        feedbacks: List[str],
    ) -> None:
        # Create app.
        app = LoadTestApp(data_filename)
        tru_app = TruApp(
            app,
            app_name="LOAD_TEST_APP",
            app_version="V1",
            connector=self._tru_session.connector,
        )
        # Load data.
        input_data = pd.read_csv(data_filename)
        input_data = input_data[["query"]]
        input_data = pd.concat(
            [input_data] * int(math.ceil(num_inputs / len(input_data))),
            ignore_index=True,
        )
        input_data = input_data.iloc[:num_inputs]
        # Ingest for single app/run.
        run_config = RunConfig(
            run_name="LOAD_TEST_RUN",
            description="description",
            dataset_name=data_filename,
            source_type="DATAFRAME",
            label="label",
            dataset_spec={"input": "query"},
        )
        run = tru_app.add_run(run_config=run_config)
        run.start(input_df=input_data)
        self._wait_for_run_to_finish(run)
        self._tru_session.force_flush()
        run.compute_metrics(feedbacks)
        self._wait_for_run_to_finish(run)
        self._tru_session.force_flush()
        # Get all events associated with this.
        NUM_SPANS_FOR_APP_INSTRUMENTATION = 4
        NUM_SPANS_FOR_FEEDBACK = 10
        NUM_SPANS_PER_INVOCATION = (
            NUM_SPANS_FOR_APP_INSTRUMENTATION + NUM_SPANS_FOR_FEEDBACK
        )
        events = self._wait_for_events(
            NUM_SPANS_PER_INVOCATION * num_inputs,
            app_name="LOAD_TEST_APP",
            return_events=True,
        )
        self._sort_events_by_record_id(events)
        # Ingest the remaining data.
        for app_idx in range(num_apps):
            for run_idx in range(num_runs):
                app_name = f"APP_{app_idx}"
                run_name = f"RUN_{run_idx}"
                self._ingest_app_and_run(app, app_name, run_config, run_name)
                self._update_events(events, app_name, run_name)
                self._ingest_events(events)
        for app_idx in range(num_apps):
            self._wait_for_events(
                NUM_SPANS_PER_INVOCATION * num_inputs * num_runs,
                app_name=f"APP_{app_idx}",
                return_events=False,
            )

    def test_ingest_data(self) -> None:
        self._test_ingest_data(
            data_filename="./tests/load/data/test_snowflake_load_test_app_data.csv",
            num_apps=5,
            num_runs=5,
            num_inputs=20,
            feedbacks=[
                "coherence",
                "answer_relevance",
                "context_relevance",
                "groundedness",
            ],
        )
        logger.info(f"SCHEMA: {self._schema}")

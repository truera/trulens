import uuid

import pandas as pd
from trulens.apps.app import TruApp
from trulens.core import Metric
from trulens.core import Selector
from trulens.core import TruSession
from trulens.core.enums import Mode
from trulens.core.run import RunConfig
from trulens.core.run import RunStatus

from tests.util.otel_test_case import OtelTestCase


class BenchmarkRows:
    pass


def toy_answer_relevance(prompt: str, response: str) -> float:
    return 1.0 if "2.3 billion" in response else 0.0


class TestOssLogIngestionRun(OtelTestCase):
    def test_log_ingestion_run_computes_client_side_metrics(self) -> None:
        session = TruSession()
        app_version = f"v_{uuid.uuid4().hex[:8]}"
        run_name = f"log_ingestion_smoke_{uuid.uuid4().hex[:8]}"

        tru_app = TruApp(
            BenchmarkRows(),
            app_name="log_ingestion_smoke_app",
            app_version=app_version,
            connector=session.connector,
            start_evaluator=False,
        )

        input_df = pd.DataFrame([
            {
                "input_id": "finance-answer-001",
                "input": "What was ACME Corp's total revenue in FY2024?",
                "output": (
                    "ACME Corp reported total revenue of 2.3 billion "
                    "in FY2024."
                ),
                "retrieved_contexts": [],
            },
            {
                "input_id": "finance-answer-002",
                "input": "What was ACME Corp's total revenue in FY2024?",
                "output": "ACME Corp was founded in Denver.",
                "retrieved_contexts": [],
            },
        ])

        metric = Metric(
            implementation=toy_answer_relevance,
            name="toy_answer_relevance",
            selectors={
                "prompt": Selector.select_record_input(),
                "response": Selector.select_record_output(),
            },
        )

        run = tru_app.add_run(
            RunConfig(
                run_name=run_name,
                dataset_name="log_ingestion_smoke_dataset",
                source_type="DATAFRAME",
                mode=Mode.LOG_INGESTION,
                dataset_spec={
                    "input_id": "input_id",
                    "record_root.input": "input",
                    "record_root.output": "output",
                    "retrieval.retrieved_contexts": "retrieved_contexts",
                },
                metric_max_workers=2,
            )
        )

        run.start(input_df=input_df)
        self.assertEqual(run.get_status(), RunStatus.INVOCATION_COMPLETED)

        self.assertEqual(
            run.compute_metrics(metrics=[metric]),
            "Metrics computation in progress.",
        )

        details = run.get_record_details()
        scores_by_input_id = dict(
            zip(details["input_id"], details["toy_answer_relevance"])
        )

        self.assertEqual(scores_by_input_id["finance-answer-001"], 1.0)
        self.assertEqual(scores_by_input_id["finance-answer-002"], 0.0)

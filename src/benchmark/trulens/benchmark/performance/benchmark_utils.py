import cProfile
import pstats
from typing import Any, Optional, Sequence

from trulens.apps.custom import TruCustomApp
from trulens.apps.custom import instrument
from trulens.core._utils import optional as optional_utils
from trulens.core.database.connector import DBConnector
from trulens.core.schema.app import RecordIngestMode
from trulens.core.schema.feedback import FeedbackDefinition
from trulens.core.session import TruSession
from trulens.core.utils import imports as import_utils

tqdm = None
with import_utils.OptionalImports(messages=optional_utils.REQUIREMENT_TQDM):
    from tqdm import tqdm


class DummyRAG:
    def retrieve(self, query: str) -> list:
        return [
            "This is a dummy context.",
            "This is another dummy context.",
            "This is a third dummy context.",
        ]

    def generate_completion(self, query: str, context_str: list) -> str:
        if len(context_str) == 0:
            return "Sorry, I couldn't find an answer to your question."

        return "This is a dummy completion with a dummy answer."

    def query(self, query: str) -> str:
        context_str = self.retrieve(query=query)
        completion = self.generate_completion(
            query=query, context_str=context_str
        )
        return completion

    def __call__(self, query: str) -> str:
        return self.query(query)


class InstrumentedDummyRag(DummyRAG):
    @instrument
    def retrieve(self, query: str) -> list:
        return super().retrieve(query)

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        return super().generate_completion(query, context_str)

    @instrument
    def query(self, query: str) -> str:
        return super().query(query)


def run_ingestion_benchmark(
    custom_app: Any,
    app_version: str,
    app_inputs: Sequence[str],
    connector: Optional[DBConnector] = None,
    app_name: str = "performance_benchmark",
    record_ingest_mode: RecordIngestMode = RecordIngestMode.IMMEDIATE,
    feedbacks: Optional[Sequence[FeedbackDefinition]] = None,
    profile_save_path: Optional[str] = None,
):
    """Runs a benchmark on the given custom app.

    Args:
        custom_app (Any): The custom app to benchmark.
        app_version (str): A name of the app version.
        app_inputs (Sequence[str]): A list of queries to benchmark.
        connector (Optional[DBConnector], optional): The Database Connector to use. If not provided, will use a local sqlite database. Defaults to None.
        app_name (str, optional): The name of the app used during benchmarking. Defaults to "performance_benchmark".
        record_ingest_mode (RecordIngestMode, optional): Specify the record ingest mode to use during benchmarking. Defaults to RecordIngestMode.IMMEDIATE.
    """
    session = TruSession(connector=connector)
    tru_app = TruCustomApp(
        custom_app,
        app_name=app_name,
        app_version=app_version,
        record_ingest_mode=record_ingest_mode,
        feedbacks=feedbacks,
    )

    with tru_app, cProfile.Profile() as pr:
        benchmark_task(app_inputs=app_inputs, custom_app=custom_app)

    pr.print_stats(sort="cumulative")
    if profile_save_path:
        stats = pstats.Stats(pr)
        stats.strip_dirs()
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.dump_stats(profile_save_path)

    session.delete_singleton()


def benchmark_task(app_inputs: Sequence[str], custom_app: Any):
    if tqdm:
        app_inputs = tqdm(app_inputs)
    for query in app_inputs:
        custom_app(query)

import random
import sys
import time

from examples.expositional.end2end_apps.custom_app.dummy import Dummy

from trulens_eval.schema import record as mod_record_schema
from trulens_eval.trace import span as mod_span
from trulens_eval.tru_custom_app import instrument


class CustomRetriever(Dummy):
    """Fake retriever."""

    def __init__(self, *args, num_contexts: int = 2, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_contexts = num_contexts

    @instrument
    def retrieve_chunks(self, data):
        """Fake chunk retrieval."""

        # Fake delay.
        self.dummy_wait()

        # Fake memory usage.
        temporary = self.dummy_allocate()

        return ([
            f"Relevant chunk: {data.upper()}",
            f"Relevant chunk: {data[::-1] * 3}",
            f"Relevant chunk: I allocated {sys.getsizeof(temporary)} bytes to pretend I'm doing something."
        ] * 3)[:self.num_contexts]

    @retrieve_chunks.is_span(
        span_type=mod_span.SpanRetriever
    ) # can also use mod_span.SpanType.RETRIEVER here
    def update_span(
        self,
        call: mod_record_schema.RecordAppCall,
        span: mod_span.SpanRetriever
    ):
        """Fill in span information from a recorded call to retrieve_chunks."""

        span.num_contexts = self.num_contexts

        span.query_text = call.args['data']

        span.query_embedding = [random.random() for _ in range(16)]

        span.distance_type = "cosine-dummy-version"

        span.retrieved_contexts = call.rets

        span.retrieved_scores = [random.random() for _ in span.retrieved_contexts]

        span.retrieved_embeddings = [
            [random.random() for _ in range(16)] for _ in span.retrieved_contexts
        ]
        
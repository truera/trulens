import sys
import time

from trulens_eval.schema import record as mod_record_schema
from trulens_eval.trace import span as mod_span
from trulens_eval.tru_custom_app import instrument


class CustomRetriever:
    """Fake retriever."""

    def __init__(self, delay: float = 0.015, alloc: int = 1024 * 1024):
        self.delay = delay
        self.alloc = alloc

    @instrument
    def retrieve_chunks(self, data):
        """Fake chunk retrieval."""

        temporary = [0x42] * self.alloc

        if self.delay > 0.0:
            time.sleep(self.delay)

        return [
            f"Relevant chunk: {data.upper()}", f"Relevant chunk: {data[::-1]}",
            f"Relevant chunk: I allocated {sys.getsizeof(temporary)} bytes to pretend I'm doing something."
        ]

    @retrieve_chunks.is_span(
        span_type=mod_span.SpanRetriever
    ) # can also use mod_span.SpanType.RETRIEVER here
    @staticmethod
    def update_span(
        call: mod_record_schema.RecordAppCall,
        span: mod_span.SpanRetriever
    ):
        """Fill in span information from a recorded call to retrieve_chunks."""

        span.input_text = call.args['data']
        span.retrieved_contexts = call.rets

import sys
import time

from examples.expositional.end2end_apps.custom_app.dummy import Dummy

from trulens_eval.schema import record as mod_record_schema
from trulens_eval.trace import span as mod_span
from trulens_eval.tru_custom_app import instrument


class CustomMemory(Dummy):
    """Dummy memory implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory_type = "dummy-memory"

        # Fake memory allocation:
        self.temporary = self.dummy_allocate()

        self.messages = []

    @instrument
    def remember(self, data: str):
        """Add a piece of data to memory."""

        # Fake delay.
        self.dummy_wait()

        self.messages.append(
            data +
            f" and I'm keeping around {sys.getsizeof(self.temporary)} bytes"
        )
    @remember.is_span(span_type=mod_span.SpanMemory)
    def set_memory_span(
        self,
        call: mod_record_schema.RecordAppCall,
        span: mod_span.SpanMemory
    ):
        """Fill in dummy memory span info based on the above call."""

        span.memory_type = self.memory_type

        span.remembered = call.args['data']
    
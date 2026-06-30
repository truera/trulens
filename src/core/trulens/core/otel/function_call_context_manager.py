from typing import Union
import uuid

from opentelemetry import trace
from opentelemetry.baggage import get_baggage
from opentelemetry.baggage import remove_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from opentelemetry.trace import get_current_span
from opentelemetry.trace.span import Span
from trulens.experimental.otel_tracing.core.session import TRULENS_SERVICE_NAME
from trulens.otel.semconv.trace import SpanAttributes


class UseCurrentSpanFunctionCallContextManager:
    def __enter__(self):
        return get_current_span()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class CreateSpanFunctionCallContextManager:
    def __init__(self, span_name: str) -> None:
        self.span_name = span_name
        self.span_context_manager = None
        self.token = None
        self._started_record = False

    def __enter__(self) -> Span:
        # Set record_id into context.
        started_record = False
        nested_record_root = False

        record_id = get_baggage(SpanAttributes.RECORD_ID)
        parent_record_id = get_baggage(
            "__trulens_nested_record_parent_record_id__"
        )
        parent_span_id = get_baggage(
            "__trulens_nested_record_parent_span_id__"
        )
        parent_app_id = get_baggage(
            "__trulens_nested_record_parent_app_id__"
        )

        if (
            record_id is not None
            and parent_record_id == record_id
            and parent_span_id is not None
            and parent_app_id is not None
        ):
            nested_record_root = True

        if not record_id or nested_record_root:
            started_record = True
            record_id = str(uuid.uuid4())
            self.token = context_api.attach(
                set_baggage(SpanAttributes.RECORD_ID, record_id)
            )
            recording = get_baggage("__trulens_recording__")
            if recording is not None:
                recording.add_record_id(record_id)

        self._started_record = started_record
        # Create span.
        self.span_context_manager = (
            trace.get_tracer_provider()
            .get_tracer(TRULENS_SERVICE_NAME)
            .start_as_current_span(name=self.span_name)
        )
        span = self.span_context_manager.__enter__()
        if started_record:
            span_type = (
                SpanAttributes.SpanType.NESTED_RECORD_ROOT
                if nested_record_root
                else SpanAttributes.SpanType.RECORD_ROOT
            )
            span.set_attribute(SpanAttributes.SPAN_TYPE, span_type)
            if nested_record_root:
                span.set_attribute(
                    SpanAttributes.NESTED_RECORD_ROOT.PARENT_SPAN_ID,
                    parent_span_id,
                )
                span.set_attribute(
                    SpanAttributes.NESTED_RECORD_ROOT.PARENT_APP_ID,
                    parent_app_id,
                )
        return span

    def __exit__(self, exc_type, exc_val, exc_tb):
        e_ret = None
        # Clean up span.
        try:
            if self.span_context_manager is not None:
                self.span_context_manager.__exit__(exc_type, exc_val, exc_tb)
                self.span_context_manager = None
        except Exception as e_span:
            e_ret = e_span
        # Clean up context.
        try:
            if self.token is not None:
                remove_baggage(SpanAttributes.RECORD_ID)
                context_api.detach(self.token)
                self.token = None
        except Exception as e_context:
            e_ret = e_context if e_ret is None else e_ret
        # Throw any errors we found.
        if e_ret is not None:
            raise e_ret


def create_function_call_context_manager(
    create_new_span: bool, span_name: str
) -> Union[
    UseCurrentSpanFunctionCallContextManager,
    CreateSpanFunctionCallContextManager,
]:
    if create_new_span:
        return CreateSpanFunctionCallContextManager(span_name)
    return UseCurrentSpanFunctionCallContextManager()

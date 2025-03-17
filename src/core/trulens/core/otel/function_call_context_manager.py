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
    def __init__(self, span_name: str, is_record_root: bool):
        self.span_name = span_name
        self.is_record_root = is_record_root
        self.span_context_manager = None
        self.token = None

    def __enter__(self) -> Span:
        # Set record_id into context.
        if self.is_record_root:
            record_id = get_baggage(SpanAttributes.RECORD_ID)
            if not record_id:
                record_id = str(uuid.uuid4())
                self.token = context_api.attach(
                    set_baggage(SpanAttributes.RECORD_ID, record_id)
                )
        # Create span.
        self.span_context_manager = (
            trace.get_tracer_provider()
            .get_tracer(TRULENS_SERVICE_NAME)
            .start_as_current_span(name=self.span_name)
        )
        span = self.span_context_manager.__enter__()
        if self.is_record_root:
            span.set_attribute(
                SpanAttributes.SPAN_TYPE, SpanAttributes.SpanType.RECORD_ROOT
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
    create_new_span: bool,
    span_name: str,
    is_record_root: bool,
) -> Union[
    UseCurrentSpanFunctionCallContextManager,
    CreateSpanFunctionCallContextManager,
]:
    if create_new_span:
        return CreateSpanFunctionCallContextManager(span_name, is_record_root)
    return UseCurrentSpanFunctionCallContextManager()

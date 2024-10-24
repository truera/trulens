from __future__ import annotations

import contextvars
import time
from typing import (
    Iterable,
)

from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.utils import python as python_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing.core import trace as core_otel
from trulens.experimental.otel_tracing.core import trace as core_trace


class _App(core_app.App):
    # TODO(otel_tracing): Roll into core_app.App once no longer experimental.

    # WithInstrumentCallbacks requirement
    def get_active_contexts(
        self,
    ) -> Iterable[core_instruments._RecordingContext]:
        """Get all active recording contexts."""

        recording = self.recording_contexts.get(contextvars.Token.MISSING)

        while recording is not contextvars.Token.MISSING:
            yield recording
            recording = recording.token.old_value

    # WithInstrumentCallbacks requirement
    def _on_new_recording_span(
        self,
        recording_span: core_trace.Span,
    ):
        if self.session._experimental_otel_exporter is not None:
            # Export to otel exporter if exporter was set in workspace.
            to_export = []
            for span in recording_span.iter_family(include_phantom=True):
                if isinstance(span, core_otel.Span):
                    e_span = span.otel_freeze()
                    to_export.append(e_span)
                else:
                    print(f"Warning, span {span.name} is not exportable.")

            print(
                f"{text_utils.UNICODE_CHECK} Exporting {len(to_export)} spans to {python_utils.class_name(self.session._experimental_otel_exporter)}."
            )
            self.session._experimental_otel_exporter.export(to_export)

    # WithInstrumentCallbacks requirement
    def _on_new_root_span(
        self,
        recording: core_instruments._RecordingContext,
        root_span: core_trace.Span,
    ) -> record_schema.Record:
        tracer = root_span.context.tracer

        record = tracer.record_of_root_span(
            root_span=root_span, recording=recording
        )
        recording.records.append(record)
        # need to jsonify?

        error = root_span.error

        if error is not None:
            # May block on DB.
            self._handle_error(record=record, error=error)
            raise error

        # Will block on DB, but not on feedback evaluation, depending on
        # FeedbackMode:
        record.feedback_and_future_results = self._handle_record(record=record)
        if record.feedback_and_future_results is not None:
            record.feedback_results = [
                tup[1] for tup in record.feedback_and_future_results
            ]

        if record.feedback_and_future_results is None:
            return record

        if self.feedback_mode == feedback_schema.FeedbackMode.WITH_APP_THREAD:
            # Add the record to ones with pending feedback.

            self.records_with_pending_feedback_results.add(record)

        elif self.feedback_mode == feedback_schema.FeedbackMode.WITH_APP:
            # If in blocking mode ("WITH_APP"), wait for feedbacks to finished
            # evaluating before returning the record.

            record.wait_for_feedback_results()

        return record

    # For use as a context manager.
    def __enter__(self):
        # EXPERIMENTAL(otel_tracing): replacement to recording context manager.

        tracer: core_trace.Tracer = core_trace.trulens_tracer()

        recording_span_ctx = tracer.recording()
        recording_span: core_trace.PhantomSpanRecordingContext = (
            recording_span_ctx.__enter__()
        )
        recording = core_trace._RecordingContext(
            app=self,
            tracer=tracer,
            span=recording_span,
            span_ctx=recording_span_ctx,
        )
        recording_span.recording = recording
        recording_span._start_timestamp = time.time_ns()  # move to trace

        # recording.ctx = ctx

        token = self.recording_contexts.set(recording)
        recording.token = token

        return recording

    # For use as a context manager.
    def __exit__(self, exc_type, exc_value, exc_tb):
        # EXPERIMENTAL(otel_tracing): replacement to recording context manager.

        recording: core_trace._RecordingContext = self.recording_contexts.get()

        assert recording is not None, "Not in a tracing context."
        assert recording.tracer is not None, "Not in a tracing context."
        assert recording.span is not None, "Not in a tracing context."

        recording.span._end_timestamp = time.time_ns()  # move to trace

        self.recording_contexts.reset(recording.token)
        return recording.span_ctx.__exit__(exc_type, exc_value, exc_tb)

    # For use as an async context manager.
    async def __aenter__(self):
        # EXPERIMENTAL(otel_tracing)

        tracer: core_trace.Tracer = core_trace.trulens_tracer()

        recording_span_ctx = await tracer.arecording()
        recording_span: core_trace.PhantomSpanRecordingContext = (
            await recording_span_ctx.__aenter__()
        )
        recording = core_trace._RecordingContext(
            app=self,
            tracer=tracer,
            span=recording_span,
            span_ctx=recording_span_ctx,
        )
        recording_span.recording = recording
        recording_span.start_timestamp = time.time_ns()

        # recording.ctx = ctx

        token = self.recording_contexts.set(recording)
        recording.token = token

        return recording

    # For use as a context manager.
    async def __aexit__(self, exc_type, exc_value, exc_tb):
        # EXPERIMENTAL(otel_tracing)

        recording: core_trace._RecordingContext = self.recording_contexts.get()

        assert recording is not None, "Not in a tracing context."
        assert recording.tracer is not None, "Not in a tracing context."

        recording.span.end_timestamp = time.time_ns()

        self.recording_contexts.reset(recording.token)
        return await recording.span_ctx.__aexit__(exc_type, exc_value, exc_tb)

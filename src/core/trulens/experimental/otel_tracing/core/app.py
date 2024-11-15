"""
Adds otel_tracing specific features to the main [App][trulens.core.app.App]
class. This file is to be integrated into the main [app][trulens.core.app] once
otel_tracing graduates.

- Callback
  ([_on_new_root_span][trulens.experimental.otel_tracing.core.app._on_new_root_span])
  for when a new root span has finished tracing. This callback adds the records'
  worth of spans to the database and possibly executes feedback

- Contextmanager methods (__enter__, __exit__) for starting and stopping a
  recording context are overwritten for otel tracing to create a special
  [RecordingContextSpan][trulens.experimental.otel_tracing.core.span.RecordingContextSpan]
  span.

- Callback
  ([_on_new_recording_span][trulens.experimental.otel_tracing.core.app._on_new_recording_span])
  when the recording span (the above) is finished. This span also controls the
  exporting of spans if TruLens has been configured so.
"""

from __future__ import annotations

from typing import (
    List,
    Literal,
    Optional,
)
import weakref

from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.utils import python as python_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing.core.trace import (
    callbacks as core_callbacks,
)
from trulens.experimental.otel_tracing.core.trace import export as core_export
from trulens.experimental.otel_tracing.core.trace import span as core_span
from trulens.experimental.otel_tracing.core.trace import trace as core_trace
from trulens.semconv import trace as truconv


class _App(core_app.App):
    # TODO(otel_tracing): Roll into core_app.App once no longer experimental.

    # WithInstrumentCallbacks requirement
    def _on_new_recording_span(
        self,
        recording_span: core_span.Span,
    ):
        """Callback for when a recording span
        ([RecordingContextSpan][trulens.experimental.otel_tracing.core.span.RecordingContextSpan])
        is finished.

        Handles exporting to OTEL exporters.
        """

        exporter_ident = str(str(self.session._experimental_otel_exporter))
        if self.session._experimental_otel_exporter is not None:
            to_export: Optional[List] = []
            num_exportable = 0
        else:
            to_export = None

        for span in recording_span.iter_family():
            if to_export is not None:
                if isinstance(span, core_span.Span):
                    num_exportable += 1
                    if not core_trace.was_exported_to(
                        context=span.context,
                        to=exporter_ident,
                        mark_exported=True,
                    ):
                        e_span = span.otel_freeze()
                        to_export.append(e_span)
                else:
                    print(f"Warning, span {span.name} is not exportable.")

        if to_export is not None:
            # Export to otel exporter if exporter was set in workspace.

            print(
                f"{text_utils.UNICODE_CHECK} Exporting {len(to_export)}/{num_exportable} spans to {python_utils.class_name(self.session._experimental_otel_exporter)}."
            )
            self.session._experimental_otel_exporter.export(to_export)

    # WithInstrumentCallbacks requirement
    def _on_new_root_span(
        self,
        recording: core_instruments._RecordingContext,
        root_span: core_span.Span,
    ) -> record_schema.Record:
        """Callback for when a new trace root span
        ([LiveRecordRoot][trulens.experimental.otel_tracing.core.span.LiveRecordRoot])
        is finished.

        Controls saving spans to the database as well as feedback execution or scheduling.
        """

        record = core_export.record_of_root_span(
            root_span=root_span, recording=recording
        )
        recording.records.append(record)
        # need to jsonify?

        typed_spans = record.experimental_otel_spans

        db_ident = str(self.connector.db)

        unwritten_spans = [
            span
            for span in typed_spans
            if core_trace.was_exported_to(
                context=span.context, to=db_ident, mark_exported=True
            )
        ]

        self.connector.db.insert_spans(spans=unwritten_spans)

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

        recording_span_ctx = tracer.start_as_current_span(
            cls=core_span.RecordingContextSpan,
            name=truconv.SpanAttributes.RECORDING.SPAN_NAME_PREFIX
            + self.app_name,
            live_app=weakref.ref(self),
        )

        recording_span: core_span.RecordingContextSpan = (
            recording_span_ctx.__enter__()
        )

        recording = core_callbacks._RecordingContext(
            app=self,
            tracer=tracer,
            span=recording_span,
            span_ctx=recording_span_ctx,
        )
        recording_span.live_recording = recording

        token = self.recording_contexts.set(recording)
        recording.token = token

        return recording

    # For use as a context manager.
    def __exit__(self, exc_type, exc_value, exc_tb) -> Literal[False]:
        # EXPERIMENTAL(otel_tracing): replacement to recording context manager.

        recording: core_callbacks._RecordingContext = (
            self.recording_contexts.get()
        )

        assert recording is not None, "Not in a tracing context."
        assert recording.tracer is not None, "Not in a tracing context."
        assert recording.span is not None, "Not in a tracing context."

        self.recording_contexts.reset(recording.token)

        recording.span_ctx.__exit__(exc_type, exc_value, exc_tb)

        return False

    # For use as an async context manager.
    async def __aenter__(self):
        # EXPERIMENTAL(otel_tracing)

        tracer: core_trace.Tracer = core_trace.trulens_tracer()

        recording_span_ctx = tracer.astart_as_current_span(
            cls=core_span.RecordingContextSpan,
            name=truconv.SpanNames.RECORDING_CONTEXT_PREFIX + self.app_name,
        )
        recording_span: core_span.RecordingContextSpan = (
            await recording_span_ctx.__aenter__()
        )
        recording = core_trace._RecordingContext(
            app=self,
            tracer=tracer,
            span=recording_span,
            span_ctx=recording_span_ctx,
        )
        recording_span.live_recording = recording

        token = self.recording_contexts.set(recording)
        recording.token = token

        return recording

    # For use as a context manager.
    async def __aexit__(self, exc_type, exc_value, exc_tb) -> Literal[False]:
        # EXPERIMENTAL(otel_tracing)

        recording: core_trace._RecordingContext = self.recording_contexts.get()

        assert recording is not None, "Not in a tracing context."
        assert recording.tracer is not None, "Not in a tracing context."

        self.recording_contexts.reset(recording.token)

        await recording.span_ctx.__aexit__(exc_type, exc_value, exc_tb)

        return False

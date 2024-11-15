# ruff: noqa: E402

""" """

from __future__ import annotations

import contextvars
import inspect
import logging
import os
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
import weakref

from trulens.core.schema import record as record_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing.core._utils import wrap as wrap_utils
from trulens.experimental.otel_tracing.core.trace import context as core_context
from trulens.experimental.otel_tracing.core.trace import otel as core_otel
from trulens.experimental.otel_tracing.core.trace import span as core_span
from trulens.experimental.otel_tracing.core.trace import trace as core_trace
from trulens.semconv import trace as truconv

if TYPE_CHECKING:
    # Need to model_rebuild classes that use these:
    from trulens.experimental.otel_tracing.core.trace import span as core_trace

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")  # callable return type
E = TypeVar("E")  # iterator/generator element type
S = TypeVar("S")  # span type

INSTRUMENT: str = "__tru_instrumented"
"""Attribute name to be used to flag instrumented objects/methods/others."""

APPS: str = "__tru_apps"
"""Attribute name for storing apps that expect to be notified of calls."""


class TracingCallbacks(wrap_utils.CallableCallbacks[R], Generic[R, S]):
    """Extension of CallableCallbacks that adds tracing to the wrapped callable
    as implemented using tracer and spans."""

    def __init__(
        self,
        func_name: str,
        span_type: Type[S] = core_span.LiveSpanCall,
        enter_contexts: bool = True,
        **kwargs: Dict[str, Any],
    ):
        """
        Args:
            enter_contexts: Whether to enter the context managers in this class
                init. If a subclass needs to add more context managers before
                entering, set this flag to false in `super().__init__` and then
                call `self._enter_contexts()` in own subclass `__init__`.
        """

        super().__init__(**kwargs)

        self.func_name: str = func_name

        self.obj: Optional[object] = None
        self.obj_cls: Optional[Type] = None
        self.obj_id: Optional[int] = None

        if not issubclass(span_type, core_span.LiveSpanCall):
            raise ValueError("span_type must be a subclass of LiveSpanCall.")

        self.span_context: ContextManager[core_span.LiveSpanCall] = (
            core_trace.trulens_tracer().start_as_current_span(cls=span_type, name=truconv.SpanAttributes.CALL.SPAN_NAME_PREFIX + self.func_name)
        )
        # Will be filled in by _enter_contexts.
        self.span: Optional[core_span.LiveSpanCall] = None

        # Keeping track of possibly multiple contexts for subclasses to add
        # more.
        self.context_managers: List[ContextManager[core_span.LiveSpanCall]] = [
            self.span_context
        ]
        self.spans: List[
            core_otel.Span
        ] = []  # keep track of the spans we enter

        if enter_contexts:
            self._enter_contexts()

    def _enter_contexts(self):
        """Enter all of the context managers registered in this class.

        This includes the span for this callback but might include others if
        subclassed.
        """

        for context_manager in self.context_managers:
            span = context_manager.__enter__()
            self.spans.append(span)
            if context_manager == self.span_context:
                # Make a special note of the main span for this callback.
                self.span = span

        if self.span is None:
            raise RuntimeError("Main span was not created in this context.")

        # Propagate some fields from parent. Note that these may be updated by
        # the subclass of this callback class when new record roots get added.
        parent_span = self.span.parent_span
        if parent_span is not None:
            if isinstance(parent_span, core_span.Span):
                self.span.record_ids = parent_span.record_ids
                self.span.app_ids = parent_span.app_ids
            if isinstance(parent_span, core_span.LiveSpan):
                self.span.live_apps = parent_span.live_apps

    def _exit_contexts(self, error: Optional[Exception]) -> Optional[Exception]:
        """Exit all of the context managers registered in this class given the
        innermost context's exception optionally.

        Returns the unhandled error if the managers did not absorb it.
        """

        # Exit the contexts starting from the innermost one.
        for context_manager in self.context_managers[::-1]:
            if error is not None:
                try:
                    if context_manager.__exit__(
                        type(error), error, error.__traceback__
                    ):
                        # If the context absorbed the error, we don't propagate the
                        # error to outer contexts.
                        error = None

                except Exception as next_error:
                    # Manager might have absorbed the error but raised another
                    # so this error may not be the same as the original. While
                    # python docs say not to do this, it may happen due to bad
                    # exit implementation or just people not following the spec.
                    error = next_error

            else:
                context_manager.__exit__(None, None, None)

        return error

    def on_callable_call(
        self, bound_arguments: inspect.BoundArguments, **kwargs: Dict[str, Any]
    ) -> inspect.BoundArguments:
        temp = super().on_callable_call(
            bound_arguments=bound_arguments, **kwargs
        )

        if "self" in bound_arguments.arguments:
            # TODO: need some generalization
            self.obj = bound_arguments.arguments["self"]
            self.obj_cls = type(self.obj)
            self.obj_id = id(self.obj)
        else:
            logger.warning("No self in bindings for %s.", self)

        span = self.span

        assert span is not None, "Contexts not yet entered."
        span.process_id = os.getpid()
        span.thread_id = threading.get_native_id()

        return temp

    def on_callable_end(self):
        super().on_callable_end()

        error = None
        try:
            error = self._exit_contexts(self.error)

        except Exception as e:
            # Just in case exit contexts raises another error
            error = e

        finally:
            span = self.span
            if span is None:
                raise RuntimeError("Contexts not yet entered.")

            # LiveSpanCall attributes
            span.call_id = self.call_id
            span.live_obj = self.obj
            span.live_cls = self.obj_cls
            span.live_func = self.func
            span.live_args = self.call_args
            span.live_kwargs = self.call_kwargs
            span.live_bound_arguments = self.bound_arguments
            span.live_sig = self.sig
            span.live_ret = self.ret
            span.live_error = error


class _RecordingContext:
    """Manager of the creation of records from record calls.

    An instance of this class is produced when using an
    [App][trulens_eval.app.App] as a context mananger, i.e.:
    Example:
        ```python
        app = ...  # your app
        truapp: TruChain = TruChain(app, ...) # recorder for LangChain apps
        with truapp as recorder:
            app.invoke(...) # use your app
        recorder: RecordingContext
        ```

    Each instance of this class produces a record for every "root" instrumented
    method called. Root method here means the first instrumented method in a
    call stack. Note that there may be more than one of these contexts in play
    at the same time due to:
    - More than one wrapper of the same app.
    - More than one context manager ("with" statement) surrounding calls to the
      same app.
    - Calls to "with_record" on methods that themselves contain recording.
    - Calls to apps that use trulens internally to track records in any of the
      supported ways.
    - Combinations of the above.
    """

    def __init__(
        self,
        app: _WithInstrumentCallbacks,
        record_metadata: serial_utils.JSON = None,
        tracer: Optional[core_trace.Tracer] = None,
        span: Optional[core_span.RecordingContextSpan] = None,
        span_ctx: Optional[core_context.SpanContext] = None,
    ):
        self.calls: Dict[types_schema.CallID, record_schema.RecordAppCall] = {}
        """A record (in terms of its RecordAppCall) in process of being created.

        Storing as a map as we want to override calls with the same id which may
        happen due to methods producing awaitables or generators. These result
        in calls before the awaitables are awaited and then get updated after
        the result is ready.
        """
        # TODEP: To deprecated after migration to span-based tracing.

        self.records: List[record_schema.Record] = []
        """Completed records."""

        self.lock: threading.Lock = threading.Lock()
        """Lock blocking access to `records` when adding calls or
        finishing a record."""

        self.token: Optional[contextvars.Token] = None
        """Token for context management."""

        self.app: _WithInstrumentCallbacks = app
        """App for which we are recording."""

        self.record_metadata = record_metadata
        """Metadata to attach to all records produced in this context."""

        self.tracer: Optional[core_trace.Tracer] = tracer
        """EXPERIMENTAL(otel_tracing): OTEL-like tracer for recording.
        """

        self.span: Optional[core_span.RecordingContextSpan] = span
        """EXPERIMENTAL(otel_tracing): Span that represents a recording context
        (the with block)."""

        self.span_ctx = span_ctx
        """EXPERIMENTAL(otel_tracing): The context manager for the above span.
        """

    @property
    def spans(self) -> Dict[core_context.SpanContext, core_otel.Span]:
        """EXPERIMENTAL(otel_tracing): Get the spans of the tracer in this context."""

        if self.tracer is None:
            return {}

        return self.tracer.spans

    def __iter__(self):
        return iter(self.records)

    def get(self) -> record_schema.Record:
        """Get the single record only if there was exactly one or throw
        an error otherwise."""

        if len(self.records) == 0:
            raise RuntimeError("Recording context did not record any records.")

        if len(self.records) > 1:
            raise RuntimeError(
                "Recording context recorded more than 1 record. "
                "You can get them with ctx.records, ctx[i], or `for r in ctx: ...`."
            )

        return self.records[0]

    def __getitem__(self, idx: int) -> record_schema.Record:
        return self.records[idx]

    def __len__(self):
        return len(self.records)

    def __hash__(self) -> int:
        # The same app can have multiple recording contexts.
        return hash(id(self.app)) + hash(id(self.records))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def add_call(self, call: record_schema.RecordAppCall):
        """Add the given call to the currently tracked call list."""
        # TODEP: To deprecated after migration to span-based tracing.

        with self.lock:
            # NOTE: This might override existing call record which happens when
            # processing calls with awaitable or generator results.
            self.calls[call.call_id] = call

    def finish_record(
        self,
        calls_to_record: Callable[
            [
                List[record_schema.RecordAppCall],
                types_schema.Metadata,
                Optional[record_schema.Record],
            ],
            record_schema.Record,
        ],
        existing_record: Optional[record_schema.Record] = None,
    ):
        """Run the given function to build a record from the tracked calls and any
        pre-specified metadata."""
        # TODEP: To deprecated after migration to span-based tracing.

        with self.lock:
            record = calls_to_record(
                list(self.calls.values()), self.record_metadata, existing_record
            )
            self.calls = {}

            if existing_record is None:
                # If existing record was given, we assume it was already
                # inserted into this list.
                self.records.append(record)

        return record


class _WithInstrumentCallbacks:
    """Abstract definition of callbacks invoked by Instrument during
    instrumentation or when instrumented methods are called.

    Needs to be mixed into [App][trulens_eval.app.App].
    """

    # Called during instrumentation.
    def on_method_instrumented(
        self, obj: object, func: Callable, path: serial_utils.Lens
    ):
        """Callback to be called by instrumentation system for every function
        requested to be instrumented.

        Given are the object of the class in which `func` belongs
        (i.e. the "self" for that function), the `func` itsels, and the `path`
        of the owner object in the app hierarchy.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).

            path: The path of the owner object in the app hierarchy.
        """

        raise NotImplementedError

    # Called during invocation.
    def get_method_path(self, obj: object, func: Callable) -> serial_utils.Lens:
        """Get the path of the instrumented function `func`, a member of the class
        of `obj` relative to this app.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).
        """

        raise NotImplementedError

    # WithInstrumentCallbacks requirement
    def get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, serial_utils.Lens]]:
        """EXPERIMENTAL(otel_tracing): Get the methods (rather the inner
        functions) matching the given `func` and the path of each.

        Args:
            func: The function to match.
        """

        raise NotImplementedError

    # Called after recording of an invocation.
    def _on_new_root_span(
        self,
        ctx: _RecordingContext,
        root_span: core_span.LiveSpanCall,
    ) -> record_schema.Record:
        """EXPERIMENTAL(otel_tracing): Called by instrumented methods if they
        are root calls (first instrumented methods in a call stack).

        Args:
            ctx: The context of the recording.

            root_span: The root span that was recorded.
        """
        # EXPERIMENTAL(otel_tracing)

        raise NotImplementedError


class AppTracingCallbacks(TracingCallbacks[R, S]):
    """Extension to TracingCallbacks that keep track of apps that are
    instrumenting their constituent calls.

    Also inserts LiveRecordRoot spans
    """

    @classmethod
    def on_callable_wrapped(
        cls,
        wrapper: Callable[..., R],
        app: _WithInstrumentCallbacks,
        **kwargs: Dict[str, Any],
    ):
        # Adds the requesting app to the list of apps the wrapper is
        # instrumented for.

        if not python_utils.safe_hasattr(wrapper, APPS):
            apps: weakref.WeakSet[_WithInstrumentCallbacks] = weakref.WeakSet()
            setattr(wrapper, APPS, apps)
        else:
            apps = python_utils.safe_getattr(wrapper, APPS)

        apps.add(app)

        return super().on_callable_wrapped(wrapper=wrapper, **kwargs)

    def __init__(
        self,
        span_type: Type[core_otel.Span] = core_span.LiveSpanCall,
        **kwargs: Dict[str, Any],
    ):
        # Do not enter the context managers in the superclass init as we need to
        # add another outer one possibly depending on the below logic.
        super().__init__(span_type=span_type, enter_contexts=False, **kwargs)

        # Get all of the apps that have instrumented this call.
        apps = python_utils.safe_getattr(self.wrapper, APPS)
        # Determine which of this apps are actually recording:
        apps = {app for app in apps if app.recording_contexts.get() is not None}

        trace_root_span_context_managers: List[ContextManager] = []

        current_span = core_trace.trulens_tracer().current_span
        record_map = {}
        started_apps: weakref.WeakSet[Any] = weakref.WeakSet()  # Any = App

        # Logic here needs to determine whether to add new RecordRoot spans. Get
        # already tracking apps/records from current (soon to be parent) span.
        if current_span is None:
            pass
        else:
            if isinstance(current_span, core_span.Span):
                record_map.update(current_span.record_ids)

            if isinstance(current_span, core_span.LiveSpan):
                started_apps = started_apps.union(current_span.live_apps)

        # Now for each app that instrumented the method that is not yet in
        # record_ids, create a span context manager for it and add it to
        # record_ids of the new created span.

        for app in set(apps).difference(started_apps):
            new_record_id = types_schema.TraceRecordID.default_py()
            record_map[app.app_id] = new_record_id
            print(
                f"{text_utils.UNICODE_CHECK} New record {new_record_id} on call to {python_utils.callable_name(self.func)} by app {app.app_name}."
            )
            started_apps.add(app)
            trace_root_span_context_managers.append(
                core_trace.trulens_tracer().start_as_current_span(
                    cls=core_span.LiveRecordRoot,
                    name=truconv.SpanAttributes.RECORD_ROOT.SPAN_NAME_PREFIX
                    + app.app_name,  # otel Span field
                    record_ids=dict(record_map),  # trulens Span field
                    app_ids={
                        app.app_id for app in started_apps
                    },  # trulens Span field
                    live_apps=weakref.WeakSet(started_apps),  # LiveSpan field
                    live_app=weakref.ref(app),  # LiveRecordRoot field
                    record_id=new_record_id,  # LiveRecordRoot field
                )
            )

        # Importantly, add the managers for the trace root `before` the span
        # managed by TracingCallbacks. This makes sure the root spans are the
        # parents of the call span. The order of root spans does not matter as
        # we stored them in a set in wrapper.APPS.
        self.context_managers = (
            trace_root_span_context_managers + self.context_managers
        )

        # Finally enter the contexts, possibly including the ones we just added.
        self._enter_contexts()

        assert self.span is not None, "Contexts not yet entered."

        # Make note of all the apps the main span is recording for and the app
        # to record map.
        if issubclass(span_type, core_span.Span):
            self.span.record_ids = record_map
            self.span.app_ids = {app.app_id for app in started_apps}

        if issubclass(span_type, core_span.LiveSpan):
            self.span.live_apps = started_apps

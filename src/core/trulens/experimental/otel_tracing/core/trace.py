# ruff: noqa: E402

"""Tracer for OTEL tracing.

Adds TruLens specific features on top of the minimal OTEL Tracer.

!!! Note
    Most of the module is EXPERIMENTAL(otel_tracing) though it includes some existing
    non-experimental classes moved here to resolve some circular import issues.
"""

from __future__ import annotations

from collections import defaultdict
import contextvars
import inspect
import logging
import os
import sys
import threading as th
from threading import Lock
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)
import weakref

from opentelemetry.util import types as types_api
import pydantic
from trulens.core.schema import base as base_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing import _feature
from trulens.experimental.otel_tracing.core import otel as core_otel
from trulens.experimental.otel_tracing.core import span as core_span
from trulens.experimental.otel_tracing.core._utils import wrap as wrap_utils
from trulens.semconv import trace as truconv

_feature._FeatureSetup.assert_optionals_installed()  # checks to make sure otel is installed

if sys.version_info < (3, 9):
    from functools import lru_cache as fn_cache
else:
    from functools import cache as fn_cache

T = TypeVar("T")
R = TypeVar("R")  # callable return type
E = TypeVar("E")  # iterator/generator element type
S = TypeVar("S")  # span type

logger = logging.getLogger(__name__)

INSTRUMENT: str = "__tru_instrumented"
"""Attribute name to be used to flag instrumented objects/methods/others."""

APPS: str = "__tru_apps"
"""Attribute name for storing apps that expect to be notified of calls."""


class Tracer(core_otel.Tracer):
    """TruLens additions on top of [OTEL Tracer][opentelemetry.trace.Tracer]."""

    # TODO: Create a Tracer that does not record anything. Can either be a
    # setting to this tracer or a separate "NullTracer". We need non-recording
    # users to not incur much overhead hence need to be able to disable most of
    # the tracing logic when appropriate.

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Overrides core_otel.Tracer._span_class
    _span_class: Type[core_span.Span] = pydantic.PrivateAttr(core_span.Span)

    @property
    def spans(self) -> Dict[core_otel.SpanContext, core_span.Span]:
        return self._tracer_provider.spans

    @property
    def current_span(self) -> Optional[core_span.Span]:
        if (context := self.current_span_context) is None:
            return None

        return self.spans.get(context)

    def start_span(self, *args, **kwargs):
        """Like OTEL start_span except also keeps track of the span just created."""

        new_span = super().start_span(*args, **kwargs)

        self.spans[new_span.context] = new_span

        return new_span

    @staticmethod
    def _fill_stacks(
        span: core_span.Span,
        get_method_path: Callable,
        span_stacks: Dict[
            core_span.Span, List[record_schema.RecordAppCallMethod]
        ],
        stack: Optional[List[record_schema.RecordAppCallMethod]] = None,
    ):
        """Populate span_stacks with a mapping of span to call stack for
        backwards compatibility with records.

        Args:
            span: Span to start from.

            get_method_path: Function that looks up lens of a given
                obj/function. This is an WithAppCallbacks method.

            span_stacks: Mapping of span to call stack. This will be modified by
                this method.

            stack: Current call stack. Recursive calls will build this up.
        """
        if stack is None:
            stack = []

        if isinstance(span, core_span.LiveSpanCall):
            if span.live_func is None:
                print(span.attributes)
                raise ValueError(f"Span {span} has no function.")

            path = get_method_path(obj=span.live_obj, func=span.live_func)

            if path is None:
                logger.warning(
                    "No path found for %s in %s.", span.live_func, span.live_obj
                )
                path = serial_utils.Lens().static

            if inspect.ismethod(span.live_func):
                # This is a method.
                frame_ident = record_schema.RecordAppCallMethod(
                    path=path,
                    method=pyschema_utils.Method.of_method(
                        span.live_func, obj=span.live_obj, cls=span.live_cls
                    ),
                )
            elif inspect.isfunction(span.live_func):
                # This is a function, not a method.
                frame_ident = record_schema.RecordAppCallMethod(
                    path=path,
                    method=None,
                    function=pyschema_utils.Function.of_function(
                        span.live_func
                    ),
                )
            else:
                raise ValueError(f"Unexpected function type: {span.live_func}")

            stack = stack + [frame_ident]
            span_stacks[span] = stack

        for subspan in span.iter_children(transitive=False):
            Tracer._fill_stacks(
                subspan,
                stack=stack,
                get_method_path=get_method_path,
                span_stacks=span_stacks,
            )

    def _call_of_spancall(
        self,
        span: core_span.LiveSpanCall,
        stack: List[record_schema.RecordAppCallMethod],
    ) -> record_schema.RecordAppCall:
        """Convert a LiveSpanCall to a RecordAppCall."""

        args = (
            dict(span.live_bound_arguments.arguments)
            if span.live_bound_arguments is not None
            else {}
        )
        if "self" in args:
            del args["self"]  # remove self

        assert span.start_timestamp is not None
        if span.end_timestamp is None:
            logger.warning(
                "Span %s has no end timestamp. It might not have yet finished recording.",
                span,
            )

        return record_schema.RecordAppCall(
            call_id=str(span.call_id),
            stack=stack,
            args={k: json_utils.jsonify(v) for k, v in args.items()},
            rets=json_utils.jsonify(span.live_ret),
            error=str(span.live_error),
            perf=base_schema.Perf(
                start_time=span.start_timestamp,
                end_time=span.end_timestamp,
            ),
            pid=span.process_id,
            tid=span.thread_id,
        )

    def record_of_root_span(
        self, recording: Any, root_span: core_span.LiveRecordRoot
    ) -> Tuple[record_schema.Record]:
        """Convert a root span to a record.

        This span has to be a call span so we can extract things like main input and output.
        """

        assert isinstance(root_span, core_span.LiveRecordRoot), type(root_span)

        # avoiding circular imports
        from trulens.experimental.otel_tracing.core import sem as core_sem

        app = recording.app

        # Use the record_id created during tracing.
        record_id = root_span.record_id

        span_stacks: Dict[
            core_span.Span, List[record_schema.RecordAppCallMethod]
        ] = {}

        self._fill_stacks(
            root_span,
            span_stacks=span_stacks,
            get_method_path=app.get_method_path,
        )

        if root_span.end_timestamp is None:
            raise RuntimeError(
                f"Root span has not finished recording: {root_span}"
            )

        root_perf = base_schema.Perf(
            start_time=root_span.start_timestamp,
            end_time=root_span.end_timestamp,
        )

        total_cost = root_span.cost_tally()

        calls = []
        spans = [core_sem.TypedSpan.semanticize(root_span)]

        root_call_span = None
        for span in root_span.iter_children():
            if isinstance(span, core_span.LiveSpanCall):
                calls.append(
                    self._call_of_spancall(span, stack=span_stacks[span])
                )
                root_call_span = root_call_span or span

            spans.append(core_sem.TypedSpan.semanticize(span))

        if root_call_span is None:
            raise ValueError("No call span found under trace root span.")

        bound_arguments = root_call_span.live_bound_arguments
        main_error = root_call_span.live_error

        if bound_arguments is not None:
            main_input = app.main_input(
                func=root_call_span.live_func,
                sig=root_call_span.live_sig,
                bindings=root_call_span.live_bound_arguments,
            )
            if main_error is None:
                main_output = app.main_output(
                    func=root_call_span.live_func,
                    sig=root_call_span.live_sig,
                    bindings=root_call_span.live_bound_arguments,
                    ret=root_call_span.live_ret,
                )
            else:
                main_output = None
        else:
            main_input = None
            main_output = None

        record = record_schema.Record(
            record_id=record_id,
            app_id=app.app_id,
            main_input=json_utils.jsonify(main_input),
            main_output=json_utils.jsonify(main_output),
            main_error=json_utils.jsonify(main_error),
            calls=calls,
            perf=root_perf,
            cost=total_cost,
            experimental_otel_spans=spans,
        )

        return record

    @staticmethod
    def find_each_child(
        span: core_span.Span, span_filter: Callable
    ) -> Iterable[core_span.Span]:
        """For each family rooted at each child of this span, find the top-most
        span that satisfies the filter."""

        for child_span in span.children_spans:
            if span_filter(child_span):
                yield child_span
            else:
                yield from Tracer.find_each_child(child_span, span_filter)

    def records_of_recording(
        self, recording: core_span.RecordingContextSpan
    ) -> Iterable[record_schema.Record]:
        """Convert a recording based on spans to a list of records."""

        for root_span in Tracer.find_each_child(
            span=recording,
            span_filter=lambda s: isinstance(s, core_span.LiveRecordRoot),
        ):
            assert isinstance(root_span, core_span.LiveRecordRoot), type(
                root_span
            )
            yield self.record_of_root_span(
                recording=recording, root_span=root_span
            )


class TracerProvider(
    core_otel.TracerProvider, metaclass=python_utils.PydanticSingletonMeta
):
    """TruLens additions on top of [OTEL TracerProvider][opentelemetry.trace.TracerProvider]."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _trace_id: types_schema.TraceID.PY_TYPE = pydantic.PrivateAttr(
        default_factory=types_schema.TraceID.default_py
    )

    def __str__(self):
        # Pydantic will not print anything useful otherwise.
        return f"{self.__module__}.{type(self).__name__}()"

    @property
    def trace_id(self) -> types_schema.TraceID.PY_TYPE:
        return self._trace_id

    # Overrides core_otel.TracerProvider._tracer_class
    _tracer_class: Type[Tracer] = pydantic.PrivateAttr(default=Tracer)

    _tracers: Dict[str, Tracer] = pydantic.PrivateAttr(default_factory=dict)

    _spans: Dict[core_otel.SpanContext, core_span.Span] = pydantic.PrivateAttr(
        default_factory=dict
    )

    @property
    def spans(self) -> Dict[core_otel.SpanContext, core_span.Span]:
        return self._spans

    _exported_map: Dict[Hashable, Set[core_otel.SpanContext]] = (
        pydantic.PrivateAttr(default_factory=lambda: defaultdict(set))
    )
    """NON-STANDARD: Each sink (hashable) is mapped to the set of span contexts
    it has received.

    This is to prevent saving the same span twice or exporting it twice. Due to
    the recording context nature of TruLens, the same spans can be processed for
    multiple apps/contexts but we don't want to write them more than once.
    """

    def was_exported_to(
        self,
        context: core_otel.SpanContext,
        to: Hashable,
        mark_exported: bool = False,
    ) -> bool:
        """Determine whether the given span context has been exported to the
        given sink.

        Optionally marks the span context as exported.
        """

        ret = context in self._exported_map[to]

        if mark_exported:
            self._exported_map[to].add(context)

        return ret

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[types_api.Attributes] = None,
    ):
        if instrumenting_module_name in self._tracers:
            return self._tracers[instrumenting_module_name]

        tracer = super().get_tracer(
            instrumenting_module_name=instrumenting_module_name,
            instrumenting_library_version=instrumenting_library_version,
            attributes=attributes,
            schema_url=schema_url,
        )

        self._tracers[instrumenting_module_name] = tracer

        return tracer


@fn_cache
def trulens_tracer_provider():
    """Global tracer provider.
    All trulens tracers are made by this provider even if a different one is
    configured for OTEL.
    """

    return TracerProvider()


def was_exported_to(
    context: core_otel.SpanContext, to: Hashable, mark_exported: bool = False
):
    """Determine whether the given span context has been exported to the given sink.

    Optionally marks the span context as exported.
    """

    return trulens_tracer_provider().was_exported_to(context, to, mark_exported)


@fn_cache
def trulens_tracer():
    from trulens.core import __version__

    return trulens_tracer_provider().get_tracer(
        instrumenting_module_name="trulens.experimental.otel_tracing.core.trace",
        instrumenting_library_version=__version__,
    )


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
            trulens_tracer().start_as_current_span(cls=span_type, name=truconv.SpanAttributes.CALL.SPAN_NAME_PREFIX + self.func_name)
        )
        # Will be filled in by _enter_contexts.
        self.span: Optional[core_span.LiveSpanCall] = None

        # Keeping track of possibly multiple contexts for subclasses to add
        # more.
        self.context_managers: List[ContextManager[core_span.LiveSpanCall]] = [
            self.span_context
        ]
        self.spans: List[
            core_span.Span
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
        span.thread_id = th.get_native_id()

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
        tracer: Optional[Tracer] = None,
        span: Optional[core_span.RecordingContextSpan] = None,
        span_ctx: Optional[core_otel.SpanContext] = None,
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

        self.lock: Lock = Lock()
        """Lock blocking access to `records` when adding calls or
        finishing a record."""

        self.token: Optional[contextvars.Token] = None
        """Token for context management."""

        self.app: _WithInstrumentCallbacks = app
        """App for which we are recording."""

        self.record_metadata = record_metadata
        """Metadata to attach to all records produced in this context."""

        self.tracer: Optional[Tracer] = tracer
        """EXPERIMENTAL(otel_tracing): OTEL-like tracer for recording.
        """

        self.span: Optional[core_span.RecordingContextSpan] = span
        """EXPERIMENTAL(otel_tracing): Span that represents a recording context
        (the with block)."""

        self.span_ctx = span_ctx
        """EXPERIMENTAL(otel_tracing): The context manager for the above span.
        """

    @property
    def spans(self) -> Dict[core_otel.SpanContext, core_span.Span]:
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
        span_type: Type[core_span.Span] = core_span.LiveSpanCall,
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

        current_span = trulens_tracer().current_span
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
                trulens_tracer().start_as_current_span(
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

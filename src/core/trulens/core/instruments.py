"""Instrumentation

This module contains the core of the app instrumentation scheme employed by
trulens to track and record apps. These details should not be relevant for
typical use cases.
"""

from __future__ import annotations

import contextvars
from contextvars import ContextVar
import dataclasses
from datetime import datetime
import functools
import inspect
from inspect import BoundArguments
from inspect import Signature
import logging
import os
import threading as th
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import weakref

import pydantic
from pydantic.v1 import BaseModel as v1BaseModel
from trulens.core import experimental as core_experimental
from trulens.core._utils.pycompat import WeakSet
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.feedback import feedback as core_feedback
from trulens.core.schema import base as base_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import imports as import_utils
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils
from trulens.experimental.otel_tracing.core.span import Attributes
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)

T = TypeVar("T")

do_not_track = ContextVar("do_not_track", default=False)


class WithInstrumentCallbacks:
    """Abstract definition of callbacks invoked by Instrument during
    instrumentation or when instrumented methods are called.

    Needs to be mixed into [App][trulens.core.app.App].
    """

    _context_contexts = contextvars.ContextVar(
        "context_contexts", default=set()
    )
    """ContextVars for storing collections of RecordingContext ."""
    _context_contexts.set(set())

    _stack_contexts = contextvars.ContextVar("stack_contexts", default={})
    """ContextVars for storing call stacks."""
    _stack_contexts.set({})

    # Called during instrumentation.
    def on_method_instrumented(
        self, obj: object, func: Callable, path: serial_utils.Lens
    ):
        """Callback to be called by instrumentation system for every function
        requested to be instrumented.

        Given are the object of the class in which `func` belongs
        (i.e. the "self" for that function), the `func` itself, and the `path`
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
        """
        Get the path of the instrumented function `func`, a member of the class
        of `obj` relative to this app.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).
        """

        raise NotImplementedError

    # WithInstrumentCallbacks requirement
    def wrap_lazy_values(
        self,
        rets: Any,
        wrap: Callable[[T], T],
        on_done: Callable[[T], T],
        context_vars: Optional[python_utils.ContextVarsOrValues],
    ) -> Any:
        """Wrap any lazy values in the return value of a method call to invoke
        handle_done when the value is ready.

        This is used to handle library-specific lazy values that are hidden in
        containers not visible otherwise. Visible lazy values like iterators,
        generators, awaitables, and async generators are handled elsewhere.

        Args:
            rets: The return value of the method call.

            wrap: A callback to be called when the lazy value is ready. Should
                return the input value or a wrapped version of it.

            on_done: Called when the lazy values is done and is no longer lazy.
                This as opposed to a lazy value that evaluates to another lazy
                values. Should return the value or wrapper.

            context_vars: The contextvars to be captured by the lazy value. If
                not given, all contexts are captured.

        Returns:
            The return value with lazy values wrapped.
        """

        if python_utils.is_lazy(rets):
            return rets

        return on_done(rets)

    # WithInstrumentCallbacks requirement
    def get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, serial_utils.Lens]]:
        """
        Get the methods (rather the inner functions) matching the given `func`
        and the path of each.

        Args:
            func: The function to match.
        """

        raise NotImplementedError

    # Called during invocation.
    def on_new_record(self, func: Callable):
        """
        Called by instrumented methods in cases where they cannot find a record
        call list in the stack. If we are inside a context manager, return a new
        call list.
        """

        raise NotImplementedError

    # Called during invocation.
    def on_add_record(
        self,
        ctx: _RecordingContext,
        func: Callable,
        sig: Signature,
        bindings: BoundArguments,
        ret: Any,
        error: Any,
        perf: base_schema.Perf,
        cost: base_schema.Cost,
        existing_record: Optional[record_schema.Record] = None,
        final: bool = True,
    ):
        """
        Called by instrumented methods if they are root calls (first
        instrumented methods in a call stack).

        Args:
            ctx: The context of the recording.

            func: The function that was called.

            sig: The signature of the function.

            bindings: The bound arguments of the function.

            ret: The return value of the function.

            error: The error raised by the function if any.

            perf: The performance of the function.

            cost: The cost of the function.

            existing_record: If the record has already been produced (i.e.
                because it was an awaitable), it can be passed here to avoid
                re-creating it.

            final: Whether this is record is final in that it is ready for
                feedback evaluation.
        """

        raise NotImplementedError


class _RecordingContext:
    """Manager of the creation of records from record calls.

    An instance of this class is produced when using an
    [App][trulens.core.app.App] as a context manager, i.e.:

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
        app: WithInstrumentCallbacks,
        record_metadata: serial_utils.JSON = None,
    ):
        self.calls: Dict[types_schema.CallID, record_schema.RecordAppCall] = {}
        """A record (in terms of its RecordAppCall) in process of being created.

        Storing as a map as we want to override calls with the same id which may
        happen due to methods producing awaitables or generators. These result
        in calls before the awaitables are awaited and then get updated after
        the result is ready.
        """

        self.records: List[record_schema.Record] = []
        """Completed records."""

        self.lock: th.Lock = th.Lock()
        """Lock blocking access to `calls` and `records` when adding calls or
        finishing a record."""

        self.token: Optional[contextvars.Token] = None
        """Token for context management."""

        self.app: weakref.ProxyType[WithInstrumentCallbacks] = weakref.proxy(
            app
        )
        """App for which we are recording."""

        self.record_metadata = record_metadata
        """Metadata to attach to all records produced in this context."""

    def __iter__(self):
        return iter(self.records)

    def get(self) -> record_schema.Record:
        """
        Get the single record only if there was exactly one. Otherwise throw an error.
        """

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
        # return id(self.app) == id(other.app) and id(self.records) == id(other.records)

    def add_call(self, call: record_schema.RecordAppCall):
        """
        Add the given call to the currently tracked call list.
        """
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
        pre-specified metadata.

        If existing_record is provided, updates that record with new data.
        """

        with self.lock:
            current_calls = dict(self.calls)  # copy
            self.calls = {}

            if existing_record is not None:
                for call in existing_record.calls:
                    current_calls[call.call_id] = call

            # Maintain an order in a record's calls to make sure the root call
            # (which returns last) is also last in the calls list:
            sorted_calls = sorted(
                current_calls.values(),
                key=lambda c: c.perf.end_time
                if c.perf is not None
                else datetime.max,
            )

            record = calls_to_record(
                sorted_calls, self.record_metadata, existing_record
            )

            if existing_record is None:
                # If existing record was given, we assume it was already
                # inserted into this list.
                self.records.append(record)

        return record


ClassFilter = Union[Type, Tuple[Type, ...]]


@dataclasses.dataclass
class InstrumentedMethod:
    method: str
    class_filter: ClassFilter
    span_type: Optional[SpanAttributes.SpanType] = None
    attributes: Attributes = None
    must_be_first_wrapper: bool = True


def class_filter_disjunction(f1: ClassFilter, f2: ClassFilter) -> ClassFilter:
    """Create a disjunction of two class filters.

    Args:
        f1: The first filter.

        f2: The second filter.
    """

    if not isinstance(f1, Tuple):
        f1 = (f1,)

    if not isinstance(f2, Tuple):
        f2 = (f2,)

    return f1 + f2


def class_filter_matches(f: ClassFilter, obj: Union[Type, object]) -> bool:
    """Check whether given object matches a class-based filter.

    A class-based filter here means either a type to match against object
    (isinstance if object is not a type or issubclass if object is a type),
    or a tuple of types to match against interpreted disjunctively.

    Args:
        f: The filter to match against.

        obj: The object to match against. If type, uses `issubclass` to
            match. If object, uses `isinstance` to match against `filters`
            of `Type` or `Tuple[Type]`.
    """

    if isinstance(f, Type):
        if isinstance(obj, Type):
            return issubclass(obj, f)

        return isinstance(obj, f)

    if isinstance(f, Tuple):
        return any(class_filter_matches(f, obj) for f in f)

    raise ValueError(f"Invalid filter {f}. Type, or a Tuple of Types expected.")


class Instrument:
    """Instrumentation tools."""

    INSTRUMENT = "__tru_instrumented"
    """Attribute name to be used to flag instrumented objects/methods/others."""

    APPS = "__tru_apps"
    """Attribute name for storing apps that expect to be notified of calls."""

    class Default:
        """Default instrumentation configuration.

        Additional components are included in subclasses of
        [Instrument][trulens.core.instruments.Instrument]."""

        MODULES = {"trulens."}
        """Modules (by full name prefix) to instrument."""

        CLASSES = set([core_feedback.Feedback])
        """Classes to instrument."""

        METHODS = [InstrumentedMethod("__call__", core_feedback.Feedback)]
        """Methods to instrument.

        Methods matching name have to pass the filter to be instrumented.
        """

        @staticmethod
        def retrieval_span(
            query_argname: str,
        ) -> Tuple[SpanAttributes.SpanType, Attributes]:
            return (
                SpanAttributes.SpanType.RETRIEVAL,
                lambda ret, exception, *args, **kwargs: {
                    SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs[query_argname],
                    SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
                        curr.page_content for curr in ret
                    ],
                },
            )

    def print_instrumentation(self) -> None:
        """Print out description of the modules, classes, methods this class
        will instrument."""

        t = "  "

        for mod in sorted(self.include_modules):
            logger.info(f"Module {mod}*")

            for cls in sorted(
                self.include_classes,
                key=lambda c: (c.__module__, c.__qualname__),
            ):
                if not cls.__module__.startswith(mod):
                    continue

                if isinstance(cls, import_utils.Dummy):
                    logger.warning(
                        f"{t * 1}Class {cls.__module__}.{cls.__qualname__}\n{t * 2}WARNING: this class could not be imported. It may have been (re)moved. Error:"
                    )
                    logger.warning(
                        text_utils.retab(
                            tab=f"{t * 3}> ", s=str(cls.original_exception)
                        )
                    )
                    continue

                logger.info(f"{t * 1}Class {cls.__module__}.{cls.__qualname__}")

                for instrumented_method in self.include_methods:
                    method = instrumented_method.method
                    class_filter = instrumented_method.class_filter
                    if class_filter_matches(
                        f=class_filter, obj=cls
                    ) and python_utils.safe_hasattr(cls, method):
                        f = getattr(cls, method)
                        logger.info(
                            f"{t * 2}Method {method}: {inspect.signature(f)}"
                        )

    def to_instrument_object(self, obj: object) -> bool:
        """Determine whether the given object should be instrumented."""

        # NOTE: some classes do not support issubclass but do support
        # isinstance. It is thus preferable to do isinstance checks when we can
        # avoid issubclass checks.
        return any(isinstance(obj, cls) for cls in self.include_classes)

    def to_instrument_class(self, cls: type) -> bool:  # class
        """Determine whether the given class should be instrumented."""

        # Sometimes issubclass is not supported so we return True just to be
        # sure we instrument that thing.

        try:
            return any(
                issubclass(cls, parent) for parent in self.include_classes
            )
        except Exception:
            return True

    def to_instrument_module(self, module_name: str) -> bool:
        """Determine whether a module with the given (full) name should be instrumented."""

        return any(
            module_name.startswith(mod2) for mod2 in self.include_modules
        )

    def __init__(
        self,
        include_modules: Optional[Iterable[str]] = None,
        include_classes: Optional[Iterable[type]] = None,
        include_methods: Optional[List[InstrumentedMethod]] = None,
        app: Optional[WithInstrumentCallbacks] = None,
    ):
        if include_modules is None:
            include_modules = []
        if include_classes is None:
            include_classes = []
        if include_methods is None:
            include_methods = []

        self.include_modules = Instrument.Default.MODULES.union(
            set(include_modules)
        )
        self.include_classes = Instrument.Default.CLASSES.union(
            set(include_classes)
        )
        self.include_methods = Instrument.Default.METHODS + include_methods

        # NOTE(piotrm): This is a weakref to prevent capturing a reference to
        # the app in method wrapper closures.
        self.app: Optional[weakref.ProxyType[WithInstrumentCallbacks]] = (
            weakref.proxy(app) if app is not None else None
        )

    @staticmethod
    def _have_context() -> bool:
        """Determine whether context vars we need for recording are available."""

        try:
            WithInstrumentCallbacks._context_contexts.get()
            WithInstrumentCallbacks._stack_contexts.get()

        except LookupError:
            logger.warning(core_endpoint._NO_CONTEXT_WARNING)
            return False

        return True

    def tracked_method_wrapper(
        self,
        query: serial_utils.Lens,
        func: Callable,
        method_name: str,
        cls: type,
        obj: object,
        span_type: Optional[SpanAttributes.SpanType] = None,
        attributes: Optional[Attributes] = None,
        must_be_first_wrapper: bool = False,
    ):
        """Wrap a method to capture its inputs/outputs/errors."""

        if self.app is None:
            raise ValueError("Instrumentation requires an app but is None.")

        if self.app.session.experimental_feature(
            core_experimental.Feature.OTEL_TRACING, freeze=True
        ):
            from trulens.core.otel.instrument import instrument

            if span_type is None:
                span_type = SpanAttributes.SpanType.UNKNOWN
            wrapper = instrument(
                span_type=span_type,
                attributes=attributes,
                must_be_first_wrapper=must_be_first_wrapper,
            )
            # return wrapper(func)?
            of_cls_method = getattr(cls, method_name)
            return wrapper(of_cls_method)

        if python_utils.safe_hasattr(func, "__func__"):
            raise ValueError("Function expected but method received.")

        if python_utils.safe_hasattr(func, Instrument.INSTRUMENT):
            logger.debug("\t\t\t%s: %s is already instrumented", query, func)

            # Notify the app instrumenting this method where it is located. Note
            # we store the method being instrumented in the attribute
            # Instrument.INSTRUMENT of the wrapped variant.
            original_func = getattr(func, Instrument.INSTRUMENT)

            self.app.on_method_instrumented(obj, original_func, path=query)

            # Add self.app, the app requesting this method to be
            # instrumented, to the list of apps expecting to be notified of
            # calls.
            existing_apps = getattr(func, Instrument.APPS)
            # The __repr__.__self__ undoes weakref.proxy .
            existing_apps.add(self.app.__repr__.__self__)  # weakref set

            return func

        # Notify the app instrumenting this method where it is located:
        self.app.on_method_instrumented(obj, func, path=query)

        logger.debug("\t\t\t%s: instrumenting %s=%s", query, method_name, func)

        sig = python_utils.safe_signature(func)

        @functools.wraps(func)
        def tru_wrapper(*args, **kwargs):
            # NOTE(piotrm): don't capture any apps in the closure of this
            # method. This method will override various instrumented functions
            # in various class definitions and will never get deallocated. If
            # there are apps in this closure, those apps will also never get
            # deallocated.

            logger.debug(
                "%s: calling instrumented sync method %s of type %s, "
                "iscoroutinefunction=%s, "
                "isasyncgeneratorfunction=%s",
                query,
                func,
                type(func),
                python_utils.is_really_coroutinefunction(func),
                inspect.isasyncgenfunction(func),
            )

            if do_not_track.get():
                return func(*args, **kwargs)

            apps = getattr(tru_wrapper, Instrument.APPS)  # weakref

            if len(apps) > 0 and not Instrument._have_context():
                return func(*args, **kwargs)

            # If not within a root method, call the wrapped function without
            # any recording.

            # Get any contexts already known from higher in the call stack.
            contexts = set(
                WithInstrumentCallbacks._context_contexts.get()
            )  # copy

            # And add any new contexts from all apps wishing to record this
            # function. This may produce some of the same contexts that were
            # already being tracked which is ok. Importantly, this might produce
            # contexts for apps that didn't instrument a method higher in the
            # call stack hence this might be the first time they are seeing an
            # instrumented method being called.
            for app in apps:
                for ctx in app.on_new_record(func):
                    contexts.add(ctx)

            if len(contexts) == 0:
                # If no app wants this call recorded, run and return without
                # instrumentation.
                logger.debug(
                    "%s: no record found or requested, not recording.", query
                )

                return func(*args, **kwargs)

            else:
                pass

            context_token = WithInstrumentCallbacks._context_contexts.set(
                contexts
            )

            # If a wrapped method was called in this call stack, get the prior
            # calls from this variable. Otherwise create a new chain stack. As
            # another wrinkle, the addresses of methods in the stack may vary
            # from app to app that are watching this method. Hence we index the
            # stacks by id of the call record list which is unique to each app.
            stacks = dict(WithInstrumentCallbacks._stack_contexts.get())  # copy

            # My own stacks to be looked up by further subcalls by the logic
            # right above. We make a copy here since we need subcalls to access
            # it but we don't want them to modify it.
            stacks_token = WithInstrumentCallbacks._stack_contexts.set(stacks)

            error = None
            rets = None
            start_time = None
            bindings = None

            # Prepare stacks with call information of this wrapped method so
            # subsequent (inner) calls will see it. For every root_method in the
            # call stack, we make a call record to add to the existing list
            # found in the stack. Path stored in `query` of this method may
            # differ between apps that use it so we have to create a separate
            # frame identifier for each, and therefore the stack. We also need
            # to use a different stack for the same reason. We index the stack
            # in `stacks` via id of the (unique) list `record`.

            # First prepare the stacks for each context.
            for ctx in contexts:
                # Get app that has instrumented this method.
                app = ctx.app

                # The path to this method according to the app.
                path = app.get_method_path(
                    args[0], func
                )  # hopefully args[0] is self, owner of func

                if path is None:
                    logger.warning(
                        "App of type %s no longer knows about object %s method %s. "
                        "Something might be going wrong.",
                        python_utils.class_name(type(app)),
                        python_utils.id_str(args[0]),
                        python_utils.callable_name(func),
                    )
                    continue

                if ctx not in stacks:
                    # If we are the first instrumented method in the chain
                    # stack, make a new stack tuple for subsequent deeper calls
                    # (if any) to look up.
                    stack = ()
                else:
                    stack = stacks[ctx]

                frame_ident = record_schema.RecordAppCallMethod(
                    path=path,
                    method=pyschema_utils.Method.of_method(
                        func, obj=args[0], cls=cls
                    ),  # important: don't use obj here as that would capture obj in closure of this wrapper
                )

                stack = stack + (frame_ident,)

                stacks[ctx] = stack  # for deeper calls to get

            # Now we will call the wrapped method. We only do so once.

            # Start of run wrapped block.
            start_time = datetime.now()

            # Create a unique call_id for this method call. This will be the
            # same across everyone Record or RecordAppCall that refers to this
            # method invocation.
            call_id = types_schema.new_call_id()

            error_str = None

            tally = None

            # Capture the current values of the relevant context vars before we
            # reset them.
            context_vars = {
                WithInstrumentCallbacks._context_contexts: WithInstrumentCallbacks._context_contexts.get(),
                WithInstrumentCallbacks._stack_contexts: WithInstrumentCallbacks._stack_contexts.get(),
            }

            try:
                # Using sig bind here so we can produce a list of key-value
                # pairs even if positional arguments were provided.
                bindings: BoundArguments = sig.bind(*args, **kwargs)

                logger.info(f"calling {func} with {args}")

                rets, tally = core_endpoint.Endpoint.track_all_costs_tally(
                    func, *args, **kwargs
                )

            except BaseException as e:
                error = e
                error_str = str(e)

                logger.error(
                    "Error calling wrapped function %s.",
                    python_utils.callable_name(func),
                )
                logger.error(traceback.format_exc())

            # Done running the wrapped function. Lets collect the results.
            # Create common information across all records.

            # Don't include self in the recorded arguments.
            nonself = {
                k: json_utils.jsonify(v)
                for k, v in (
                    bindings.arguments.items() if bindings is not None else {}
                )
                if k
                not in ["self", "_self"]  # llama_index uses "_self" sometimes
            }

            records = {}

            if app is None:
                raise ValueError("Instrumentation requires an app but is None.")

            # We implicitly assume that all apps that have instrumented this
            # method are of the same type. We use some classmethods of that
            # app type below.

            def update_call_info(rets, final=True):
                """Notify the app of the call and let it update the record if
                needed.

                `final` is passed to the new record handler in the app and tells
                it whether to the record is ready to run feedback.
                """

                # (re) generate end_time here because cases where the
                # initial end_time was just to produce an awaitable/lazy
                # before being awaited. If a stream is used, this will
                # update the end_time to the last chunk.
                end_time = datetime.now()

                record_app_args = dict(
                    call_id=call_id,
                    args=nonself,
                    perf=base_schema.Perf(
                        start_time=start_time, end_time=end_time
                    ),
                    pid=os.getpid(),
                    tid=th.get_native_id(),
                    rets=json_utils.jsonify(rets),
                    error=error_str if error is not None else None,
                )
                # End of run wrapped block.

                # Now record calls to each context.
                for ctx in contexts:
                    stack = stacks[ctx]

                    # Note that only the stack differs between each of the records in this loop.
                    record_app_args["stack"] = stack
                    call = record_schema.RecordAppCall(**record_app_args)
                    ctx.add_call(call)

                    # If stack has only 1 thing on it, we are looking at a "root
                    # call". Create a record of the result and notify the app:

                    existing_record = records.get(ctx, None)

                    if tally is None:
                        cost = base_schema.Cost()
                    else:
                        cost = tally()  # get updated cost

                    if len(stack) == 1 or existing_record is not None:
                        # If this is a root call, notify app to add the completed record
                        # into its containers:

                        records[ctx] = ctx.app.on_add_record(
                            ctx=ctx,
                            func=func,
                            sig=sig,
                            bindings=bindings,
                            ret=rets,
                            error=error,
                            perf=base_schema.Perf(
                                start_time=start_time, end_time=end_time
                            ),
                            cost=cost,
                            existing_record=existing_record,
                            final=final,
                        )

                if error is not None:
                    raise error

                return rets

            def rewrap(rets):
                """Wrap any lazy return values with handlers to update
                call/record when ready."""

                if python_utils.is_lazy(rets):
                    type_name = python_utils.class_name(type(rets))

                    logger.debug(
                        "This app produced a lazy response of type `%s`."
                        "This record will be updated once the response is available.",
                        type_name,
                    )

                    # Placeholder:
                    temp_rets = f"""
    The method {python_utils.callable_name(func)} produced a lazy response of type
    `{type_name}`. This record will be updated once the response is available. If
    this message persists, check that you are using the correct version of the app
    method and await and/or iterate over any results it produces.
    """

                    # Will add placeholder to the record:
                    update_call_info(temp_rets, final=False)

                    return python_utils.wrap_lazy(
                        rets,
                        wrap=None,
                        on_done=rewrap,
                        context_vars=context_vars,
                    )

                # Handle app-specific lazy values (like llama_index StreamResponse):
                # NOTE(piotrm): self.app is a weakref

                if python_utils.WRAP_LAZY:
                    for ctx in contexts:
                        rets = ctx.app.wrap_lazy_values(
                            rets,
                            wrap=None,
                            on_done=update_call_info,
                            context_vars=context_vars,
                        )
                else:
                    update_call_info(rets, final=True)

                return rets

            # HACK: disable these to debug context issues. See
            # App._set_context_vars .
            WithInstrumentCallbacks._stack_contexts.reset(stacks_token)
            WithInstrumentCallbacks._context_contexts.reset(context_token)

            return rewrap(rets)

        # Create a new set of apps expecting to be notified about calls to the
        # instrumented method. Making this a weakref set so that if the
        # recorder/app gets garbage collected, it will be evicted from this set.

        # NOTE(piotrm): __repr__.__self__ undoes weakref.proxy .
        apps = WeakSet([self.app.__repr__.__self__])

        # Indicate that the wrapper is an instrumented method so that we dont
        # further instrument it in another layer accidentally.
        setattr(tru_wrapper, Instrument.INSTRUMENT, func)
        setattr(tru_wrapper, Instrument.APPS, apps)

        return tru_wrapper

    def instrument_method(
        self, method_name: str, obj: Any, query: serial_utils.Lens
    ):
        """Instrument a method."""

        cls = type(obj)

        logger.debug("%s: instrumenting %s on obj %s", query, method_name, obj)

        for base in list(cls.__mro__):
            logger.debug(
                "\t%s: instrumenting base %s",
                query,
                python_utils.class_name(base),
            )

            if python_utils.safe_hasattr(base, method_name):
                original_fun = getattr(base, method_name)

                logger.debug(
                    "\t\t%s: instrumenting %s.%s",
                    query,
                    python_utils.class_name(base),
                    method_name,
                )
                setattr(
                    base,
                    method_name,
                    self.tracked_method_wrapper(
                        query=query,
                        func=original_fun,
                        method_name=method_name,
                        cls=base,
                        obj=obj,
                    ),
                )

    def instrument_class(self, cls):
        """Instrument the given class `cls`'s __new__ method.

        This is done so we can be aware when new instances are created and is
        needed for wrapped methods that dynamically create instances of classes
        we wish to instrument. As they will not be visible at the time we wrap
        the app, we need to pay attention to __new__ to make a note of them when
        they are created and the creator's path. This path will be used to place
        these new instances in the app json structure.
        """

        func = cls.__new__

        if python_utils.safe_hasattr(func, Instrument.INSTRUMENT):
            logger.debug(
                "Class %s __new__ is already instrumented.",
                python_utils.class_name(cls),
            )
            return

        # @functools.wraps(func)
        def wrapped_new(cls, *args, **kwargs):
            logger.debug(
                "Creating a new instance of instrumented class %s.",
                python_utils.class_name(cls),
            )
            # get deepest wrapped method here
            # get its self
            # get its path
            obj = func(cls)
            # for every tracked method, and every app, do this:
            # self.app.on_method_instrumented(obj, original_func, path=query)
            return obj

        cls.__new__ = wrapped_new

    def instrument_object(
        self, obj, query: serial_utils.Lens, done: Optional[Set[int]] = None
    ):
        """Instrument the given object `obj` and its components."""

        done = done or set()

        cls = type(obj)

        mro = list(cls.__mro__)
        # Warning: cls.__mro__ sometimes returns an object that can be iterated through only once.

        logger.debug(
            "%s: instrumenting object at %s of class %s",
            query,
            python_utils.id_str(obj),
            python_utils.class_name(cls),
        )

        if id(obj) in done:
            logger.debug("\t%s: already instrumented", query)
            return

        done.add(id(obj))

        # NOTE: We cannot instrument chain directly and have to instead
        # instrument its class. The pydantic.BaseModel does not allow instance
        # attributes that are not fields:
        # https://github.com/pydantic/pydantic/blob/11079e7e9c458c610860a5776dc398a4764d538d/pydantic/main.py#LL370C13-L370C13
        # .

        # Recursively instrument inner components
        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                if isinstance(
                    attr_value, python_utils.OpaqueWrapper
                ):  # never look past opaque wrapper
                    continue
                if any(
                    isinstance(attr_value, cls) for cls in self.include_classes
                ):
                    inner_query = query[attr_name]
                    self.instrument_object(attr_value, inner_query, done)

        for base in mro:
            # Some top part of mro() may need instrumentation here if some
            # subchains call superchains, and we want to capture the
            # intermediate steps. On the other hand we don't want to instrument
            # the very base classes such as object:
            # if not self.to_instrument_module(base.__module__):
            #    print(
            #        f"skipping base {base} because of module {base.__module__}"
            #    )
            #    continue

            try:
                if not self.to_instrument_class(base):
                    logger.debug(f"skipping base {base} because of class")
                    continue

            except Exception as e:
                # subclass check may raise exception
                logger.debug(
                    "\t\tWarning: checking whether %s should be instrumented resulted in an error: %s",
                    python_utils.module_name(base),
                    e,
                )
                # NOTE: Proceeding to instrument here as we don't want to miss
                # anything. Unsure why some llama_index subclass checks fail.

                # continue

            print(f"instrumenting {obj.__class__} for base {base}")

            for instrumented_method in self.include_methods:
                method_name = instrumented_method.method
                class_filter = instrumented_method.class_filter
                if python_utils.safe_hasattr(base, method_name):
                    if not class_filter_matches(f=class_filter, obj=obj):
                        continue

                    print("\tinstrumenting", method_name)
                    original_fun = getattr(base, method_name)

                    # Skip non-callable attributes (strings, properties, etc.)
                    if not callable(original_fun):
                        print(
                            f"\t\tskipping non-callable attribute {method_name}: {type(original_fun)}"
                        )
                        continue

                    # If an instrument class uses a decorator to wrap one of
                    # their methods, the wrapper will capture an un-instrumented
                    # version of the inner method which we may fail to
                    # instrument.
                    # @davidkurokawa:
                    #   This doesn't make any sense. This will remove a non
                    #   trulens wrapper such as for `llama-index` functions
                    #   decorated by `@dispatcher.span`. I also don't
                    #   understand the comment above so it's hard to just
                    #   fix this easily, but given non-OTEL is soon to be
                    #   deprecated, I'm not going to fix this.
                    #   I've also changed this from an `if` to a `while` so
                    #   that it removes all decorators since that would make
                    #   slightly more sense and it also fixes a test that
                    #   had issues due to the non-OTEL flow trying to wrap a
                    #   function with a `@wrapt.decorator` decorator which is
                    #   incompatible with the way `@functools.wraps` works.
                    original_fun = inspect.unwrap(original_fun)

                    # Sometimes the base class may be in some module but when a
                    # method is looked up from it, it actually comes from some
                    # other, even baser class which might come from builtins
                    # which we want to skip instrumenting.
                    if python_utils.safe_hasattr(original_fun, "__self__"):
                        if not self.to_instrument_module(
                            original_fun.__self__.__class__.__module__
                        ):
                            continue
                    else:
                        # Determine module here somehow.
                        pass

                    logger.debug("\t\t%s: instrumenting %s", query, method_name)

                    setattr(
                        base,
                        method_name,
                        self.tracked_method_wrapper(
                            query=query,
                            func=original_fun,
                            method_name=method_name,
                            cls=base,
                            obj=obj,
                            span_type=instrumented_method.span_type,
                            attributes=instrumented_method.attributes,
                            must_be_first_wrapper=instrumented_method.must_be_first_wrapper,
                        ),
                    )

        if self.to_instrument_object(obj) or isinstance(
            obj, (dict, list, tuple)
        ):
            vals = None
            if isinstance(obj, dict):
                attrs = obj.keys()
                vals = obj.values()

            if isinstance(obj, pydantic.BaseModel):
                # NOTE(piotrm): This will not include private fields like
                # llama_index's LLMPredictor._llm which might be useful to
                # include:
                attrs = type(obj).model_fields.keys()

            if isinstance(obj, v1BaseModel):
                attrs = obj.__fields__.keys()

            elif dataclasses.is_dataclass(type(obj)):
                attrs = (f.name for f in dataclasses.fields(obj))

            else:
                # If an object is not a recognized container type, we check that it
                # is meant to be instrumented and if so, we  walk over it manually.
                # NOTE: some llama_index objects are using dataclasses_json but most do
                # not so this section applies.
                attrs = pyschema_utils.clean_attributes(
                    obj, include_props=True
                ).keys()

            if vals is None:
                vals = [
                    python_utils.safe_getattr(obj, k, get_prop=True)
                    for k in attrs
                ]

            for k, v in zip(attrs, vals):
                if isinstance(v, (str, bool, int, float)):
                    pass

                elif self.to_instrument_module(type(v).__module__):
                    self.instrument_object(obj=v, query=query[k], done=done)

                elif isinstance(v, Sequence):
                    for i, sv in enumerate(v):
                        if self.to_instrument_class(type(sv)):
                            self.instrument_object(
                                obj=sv, query=query[k][i], done=done
                            )

                elif isinstance(v, Dict):
                    for k2, sv in v.items():
                        # Skip keys that aren't valid for Lens (like TypedDict metaclasses)
                        if not isinstance(k2, (str, int)):
                            continue
                        subquery = query[k][k2]
                        if self.to_instrument_class(type(sv)):
                            self.instrument_object(
                                obj=sv, query=subquery, done=done
                            )

                else:
                    pass

                # TODO: check if we want to instrument anything in langchain not
                # accessible through model_fields .

        else:
            logger.debug(
                "%s: Do not know how to instrument object of type %s.",
                query,
                python_utils.class_name(cls),
            )

        # Check whether bound methods are instrumented properly.


class AddInstruments:
    """Utilities for adding more things to default instrumentation filters."""

    @classmethod
    def method(
        cls,
        of_cls: type,
        name: str,
        *,
        span_type: Optional[SpanAttributes.SpanType] = None,
    ) -> None:
        """Add the class with a method named `name`, its module, and the method
        `name` to the Default instrumentation walk filters."""

        logger.debug("adding method", of_cls, name, of_cls.__module__)

        Instrument.Default.MODULES.add(of_cls.__module__)
        Instrument.Default.CLASSES.add(of_cls)
        Instrument.Default.METHODS.append(
            InstrumentedMethod(
                method=name,
                class_filter=of_cls,
                span_type=span_type,
            )
        )

    @classmethod
    def methods(cls, of_cls: type, names: Iterable[str]) -> None:
        """Add the class with methods named `names`, its module, and the named
        methods to the Default instrumentation walk filters."""

        for name in names:
            cls.method(of_cls, name)


class instrument(AddInstruments):
    """Decorator for marking methods to be instrumented in custom classes that are wrapped by App."""

    # NOTE(piotrm): Approach taken from:
    # https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class

    def __init__(self, func: Callable):
        logger.debug("decorating", func)
        self.func = func

    def __set_name__(self, cls: type, name: str):
        """
        For use as method decorator.
        """

        # Important: do this first:
        setattr(cls, name, self.func)

        if self.func is not getattr(cls, name):
            print(
                "Warning. Method to be instrumented does not belong to a class. It may not be instrumented."
            )

        # Note that this does not actually change the method, just adds it to
        # list of filters.
        self.method(cls, name)

    def __call__(_self, *args, **kwargs):
        # `_self` is used to avoid conflicts where `self` may be passed from the caller method
        return _self.func(*args, **kwargs)

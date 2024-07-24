from __future__ import annotations

import contextvars
import datetime
import inspect
from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
import threading
import time
from typing import (Any, Awaitable, Callable, ClassVar, Dict, Hashable,
                    Iterable, List, Optional, Sequence, Tuple, TypeVar, Union)

import pydantic

from trulens_eval import feedback as mod_feedback
from trulens_eval import instruments as mod_instruments
from trulens_eval import trace as mod_trace
from trulens_eval.schema import app as app_schema
from trulens_eval.schema import base as base_schema
from trulens_eval.schema import feedback as feedback_schema
from trulens_eval.schema import record as record_schema
from trulens_eval.utils import asynchro as asynchro_utils
from trulens_eval.utils import containers as container_utils
from trulens_eval.utils import json as json_utils
from trulens_eval.utils import pyschema as pyschema_utils
from trulens_eval.utils import python as python_utils
from trulens_eval.utils import serial as serial_utils
from trulens_eval.utils import text as text_utils

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# App component.
COMPONENT = Any

A = TypeVar("A")
T = TypeVar("T")

# Message produced when an attribute is looked up from our App but is actually
# an attribute of the enclosed app.
ATTRIBUTE_ERROR_MESSAGE = """
{class_name} has no attribute `{attribute_name}` but the wrapped app {app_class_name} does. If
you are calling a {app_class_name} method, retrieve it from that app instead of from
{class_name}. If you need to record your app's behaviour, use {class_name} as a context
manager as in this example:

```python
    app: {app_class_name} = ...  # your app
    truapp: {class_name} = {class_name}(app, ...)  # the truera recorder

    with truapp as recorder:
      result = app.{attribute_name}(...)

    record: Record = recorder.get() # get the record of the invocation if needed
```
"""


def instrumented_component_views(
    obj: object
) -> Iterable[Tuple[serial_utils.Lens, serial_utils.JSON]]:
    """
    Iterate over contents of `obj` that are annotated with the CLASS_INFO
    attribute/key. Returns tuples with the accessor/selector and the annotated
    object itself.
    """

    for q, o in serial_utils.all_objects(obj):
        if isinstance(o, pydantic.BaseModel
                     ) and pyschema_utils.CLASS_INFO in o.model_fields:
            yield q, o.model_dump()

        if isinstance(o, Dict) and pyschema_utils.CLASS_INFO in o:
            yield q, o


class App(app_schema.AppDefinition, mod_trace.WithInstrumentCallbacks,
          Hashable):
    """Base app recorder type.

    Non-serialized fields here while the serialized ones are defined in
    [AppDefinition][trulens_eval.schema.app.AppDefinition].

    This class is abstract. Use one of these concrete subclasses as appropriate:
    - [TruLlama][trulens_eval.tru_llama.TruLlama] for _LlamaIndex_ apps.
    - [TruChain][trulens_eval.tru_chain.TruChain] for _LangChain_ apps.
    - [TruRails][trulens_eval.tru_rails.TruRails] for _NeMo Guardrails_
        apps.
    - [TruVirtual][trulens_eval.tru_virtual.TruVirtual] for recording
        information about invocations of apps without access to those apps.
    - [TruCustomApp][trulens_eval.tru_custom_app.TruCustomApp] for custom
        apps. These need to be decorated to have appropriate data recorded.
    - [TruBasicApp][trulens_eval.tru_basic_app.TruBasicApp] for apps defined
        solely by a string-to-string method.
    """

    model_config: ClassVar[dict] = {
        # Tru, DB, most of the types on the excluded fields.
        'arbitrary_types_allowed': True
    }

    feedbacks: List[mod_feedback.Feedback] = pydantic.Field(
        exclude=True, default_factory=list
    )
    """Feedback functions to evaluate on each record."""

    tru: Optional[trulens_eval.tru.Tru] = pydantic.Field(
        default=None, exclude=True
    )
    """Workspace manager.
    
    If this is not povided, a singleton [Tru][trulens_eval.tru.Tru] will be made
    (if not already) and used.
    """

    db: Optional[trulens_eval.database.base.DB] = pydantic.Field(
        default=None, exclude=True
    )
    """Database interface.
    
    If this is not provided, a singleton
    [SQLAlchemyDB][trulens_eval.database.sqlalchemy.SQLAlchemyDB] will be
    made (if not already) and used.
    """

    app: Any = pydantic.Field(exclude=True)
    """The app to be recorded."""

    instrument: Optional[mod_instruments.Instrument] = pydantic.Field(
        None, exclude=True
    )
    """Instrumentation class.
    
    This is needed for serialization as it tells us which objects we want to be
    included in the json representation of this app.
    """

    recording_contexts: contextvars.ContextVar[mod_trace.RecordingContext] \
        = pydantic.Field(None, exclude=True)
    """Sequnces of records produced by the this class used as a context manager
    are stored in a RecordingContext.
    
    Using a context var so that context managers can be nested.
    """

    instrumented_methods: Dict[int, Dict[Callable, serial_utils.Lens]] = \
        pydantic.Field(exclude=True, default_factory=dict)
    """Mapping of instrumented methods (by id(.) of owner object and the
    function) to their path in this app."""

    records_with_pending_feedback_results: container_utils.BlockingSet[record_schema.Record] = \
        pydantic.Field(exclude=True, default_factory=container_utils.BlockingSet)
    """Records produced by this app which might have yet to finish
    feedback runs."""

    manage_pending_feedback_results_thread: Optional[threading.Thread] = \
        pydantic.Field(exclude=True, default=None)
    """Thread for manager of pending feedback results queue.
    
    See _manage_pending_feedback_results."""

    selector_check_warning: bool = False
    """Issue warnings when selectors are not found in the app with a placeholder
    record.
    
    If False, constructor will raise an error instead.
    """

    selector_nocheck: bool = False
    """Ignore selector checks entirely.
    
    This may be necessary if the expected record content cannot be determined
    before it is produced.
    """

    def __init__(
        self,
        tru: Optional[Tru] = None,
        feedbacks: Optional[Iterable[mod_feedback.Feedback]] = None,
        **kwargs
    ):
        if feedbacks is not None:
            feedbacks = list(feedbacks)
        else:
            feedbacks = []

        # for us:
        kwargs['tru'] = tru
        kwargs['feedbacks'] = feedbacks
        kwargs['recording_contexts'] = contextvars.ContextVar(
            "recording_contexts"
        )

        super().__init__(**kwargs)

        app = kwargs['app']
        self.app = app

        if self.instrument is not None:
            self.instrument.instrument_object(
                obj=self.app, query=feedback_schema.Select.Query().app
            )
        else:
            pass

        if self.feedback_mode == feedback_schema.FeedbackMode.WITH_APP_THREAD:
            self._start_manage_pending_feedback_results()

        self._tru_post_init()

    def __del__(self):
        # Can use to do things when this object is being garbage collected.
        pass

    def _start_manage_pending_feedback_results(self) -> None:
        """Start the thread that manages the queue of records with
        pending feedback results.
        
        This is meant to be run permentantly in a separate thread. It will
        remove records from the set `records_with_pending_feedback_results` as
        their feedback results are computed.
        """

        if self.manage_pending_feedback_results_thread is not None:
            raise RuntimeError("Manager Thread already started.")

        self.manage_pending_feedback_results_thread = threading.Thread(
            target=self._manage_pending_feedback_results,
            daemon=True  # otherwise this thread will keep parent alive
        )
        self.manage_pending_feedback_results_thread.start()

    def _manage_pending_feedback_results(self) -> None:
        """Manage the queue of records with pending feedback results.

        This is meant to be run permentantly in a separate thread. It will
        remove records from the set records_with_pending_feedback_results as
        their feedback results are computed.
        """

        while True:
            record = self.records_with_pending_feedback_results.peek()
            record.wait_for_feedback_results()
            self.records_with_pending_feedback_results.remove(record)

    def wait_for_feedback_results(
        self,
        feedback_timeout: Optional[float] = None
    ) -> List[record_schema.Record]:
        """Wait for all feedbacks functions to complete.
         
        Args:
            feedback_timeout: Timeout in seconds for waiting for feedback
                results for each feedback function. Note that this is not the
                total timeout for this entire blocking call.

        Returns:
            A list of records that have been waited on. Note a record will be
                included even if a feedback computation for it failed or
                timedout.

        This applies to all feedbacks on all records produced by this app. This
        call will block until finished and if new records are produced while
        this is running, it will include them.
        """

        records = []

        while not self.records_with_pending_feedback_results.empty():
            record = self.records_with_pending_feedback_results.pop()
            record.wait_for_feedback_results(feedback_timeout=feedback_timeout)
            records.append(record)

        return records

    @classmethod
    def select_context(cls, app: Optional[Any] = None) -> Lens:
        """
        Try to find retriever components in the given `app` and return a lens to
        access the retrieved contexts that would appear in a record were these
        components to execute.
        """
        if app is None:
            raise ValueError(
                "Could not determine context selection without `app` argument."
            )

        # Checking by module name so we don't have to try to import either
        # langchain or llama_index beforehand.
        if type(app).__module__.startswith("langchain"):
            from trulens_eval.tru_chain import TruChain
            return TruChain.select_context(app)

        if type(app).__module__.startswith("llama_index"):
            from trulens_eval.tru_llama import TruLlama
            return TruLlama.select_context(app)

        elif type(app).__module__.startswith("nemoguardrails"):
            from trulens_eval.tru_rails import TruRails
            return TruRails.select_context(app)

        else:
            raise ValueError(
                f"Could not determine context from unrecognized `app` type {type(app)}."
            )

    def __hash__(self):
        return hash(id(self))

    def _tru_post_init(self):
        """Database-related initialization and additional data checks.

        DB:
            - Insert the app into the database.

            - Insert feedback function definitions into the database.

        Checks:
            - In deferred mode, try to serialize and deserialize feedback functions.

            - Check that feedback function selectors are likely to refer to expected
                app or record components.
        
        """

        if self.tru is None:
            from trulens_eval.tru import Tru
            logger.debug("Using default tru.")
            self.tru = Tru()

        else:
            if self.feedback_mode == feedback_schema.FeedbackMode.NONE:
                logger.warning(
                    "`tru` is specified but `feedback_mode` is FeedbackMode.NONE. "
                    "No feedback evaluation and logging will occur."
                )

        if self.tru is not None:
            self.db = self.tru.db

            self.db.insert_app(app=self)

            if self.feedback_mode != feedback_schema.FeedbackMode.NONE:
                logger.debug("Inserting feedback function definitions to db.")

                for f in self.feedbacks:
                    self.db.insert_feedback_definition(f)

        else:
            if len(self.feedbacks) > 0:
                raise ValueError(
                    "Feedback logging requires `tru` to be specified."
                )

        if self.feedback_mode == feedback_schema.FeedbackMode.DEFERRED:
            for f in self.feedbacks:
                # Try to load each of the feedback implementations. Deferred
                # mode will do this but we want to fail earlier at app
                # constructor here.
                try:
                    f.implementation.load()
                except Exception as e:
                    raise Exception(
                        f"Feedback function {f} is not loadable. Cannot use DEFERRED feedback mode. {e}"
                    ) from e

        if not self.selector_nocheck:

            dummy = self.dummy_record()

            for feedback in self.feedbacks:
                feedback.check_selectors(
                    app=self,
                    # Don't have a record yet, but use an empty one for the non-call related fields.
                    record=dummy,
                    warning=self.selector_check_warning
                )

    def main_call(self, human: str) -> str:
        """If available, a single text to a single text invocation of this app."""

        if self.__class__.main_acall is not App.main_acall:
            # Use the async version if available.
            return asynchro_utils.sync(self.main_acall, human)

        raise NotImplementedError()

    async def main_acall(self, human: str) -> str:
        """If available, a single text to a single text invocation of this app."""

        if self.__class__.main_call is not App.main_call:
            logger.warning("Using synchronous version of main call.")
            # Use the sync version if available.
            return await asynchro_utils.desync(self.main_call, human)

        raise NotImplementedError()

    def _extract_content(self, value, content_keys=['content']):
        """
        Extracts the 'content' from various data types commonly used by libraries
        like OpenAI, Canopy, LiteLLM, etc. This method navigates nested data
        structures (pydantic models, dictionaries, lists) to retrieve the
        'content' field. If 'content' is not directly available, it attempts to
        extract from known structures like 'choices' in a ChatResponse. This
        standardizes extracting relevant text or data from complex API responses
        or internal data representations.
        
        Args:
            value: The input data to extract content from. Can be a pydantic
                   model, dictionary, list, or basic data type.
        
        Returns:
            The extracted content, which may be a single value, a list of values,
            or a nested structure with content extracted from all levels.
        """
        if isinstance(value, pydantic.BaseModel):
            content = getattr(value, 'content', None)
            if content is not None:
                return content

            # If 'content' is not found, check for 'choices' attribute which indicates a ChatResponse
            choices = getattr(value, 'choices', None)
            if choices is not None:
                # Extract 'content' from the 'message' attribute of each _Choice in 'choices'
                return [
                    self._extract_content(choice.message) for choice in choices
                ]

            # Recursively extract content from nested pydantic models
            return {
                k:
                    self._extract_content(v)
                    if isinstance(v, (pydantic.BaseModel, dict, list)) else v
                for k, v in value.dict().items()
            }

        elif isinstance(value, dict):
            # Check for 'content' key in the dictionary
            for key in content_keys:
                content = value.get(key)
                if content is not None:
                    return content

            # Recursively extract content from nested dictionaries
            return {
                k:
                    self._extract_content(v) if isinstance(v,
                                                           (dict, list)) else v
                for k, v in value.items()
            }

        elif isinstance(value, list):
            # Handle lists by extracting content from each item
            return [self._extract_content(item) for item in value]

        else:
            return value

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> serial_utils.JSON:
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """

        # ignore self
        all_args = list(v for k, v in bindings.arguments.items() if k != "self")

        # If there is only one string arg, it is a pretty good guess that it is
        # the main input.

        # if have only containers of length 1, find the innermost non-container
        focus = all_args

        while not isinstance(focus,
                             serial_utils.JSON_BASES) and len(focus) == 1:
            focus = focus[0]
            focus = self._extract_content(
                focus, content_keys=['content', 'input']
            )

            if not isinstance(focus, Sequence):
                logger.warning("Focus %s is not a sequence.", focus)
                break

        if isinstance(focus, serial_utils.JSON_BASES):
            return str(focus)

        # Otherwise we are not sure.
        logger.warning(
            "Unsure what the main input string is for the call to %s with args %s.",
            python_utils.callable_name(func), all_args
        )

        # After warning, just take the first item in each container until a
        # non-container is reached.
        focus = all_args
        while not isinstance(focus,
                             serial_utils.JSON_BASES) and len(focus) >= 1:
            focus = focus[0]
            focus = self._extract_content(focus)

            if not isinstance(focus, Sequence):
                logger.warning("Focus %s is not a sequence.", focus)
                break

        if isinstance(focus, serial_utils.JSON_BASES):
            return str(focus)

        logger.warning(
            "Could not determine main input/output of %s.", str(all_args)
        )

        return "Could not determine main input from " + str(all_args)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> serial_utils.JSON:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        # Use _extract_content to get the content out of the return value
        content = self._extract_content(ret, content_keys=['content', 'output'])

        if isinstance(content, str):
            return content

        if isinstance(content, float):
            return str(content)

        if isinstance(content, Dict):
            return str(next(iter(content.values()), ''))

        elif isinstance(content, Sequence):
            if len(content) > 0:
                return str(content[0])
            else:
                return "Could not determine main output from " + str(content)

        else:
            logger.warning("Could not determine main output from %s.", content)
            return str(
                content
            ) if content is not None else "Could not determine main output from " + str(
                content
            )

    # WithInstrumentCallbacks requirement
    def on_method_instrumented(
        self, obj: object, func: Callable, path: serial_utils.Lens
    ):
        """
        Called by instrumentation system for every function requested to be
        instrumented by this app.
        """

        if id(obj) in self.instrumented_methods:

            funcs = self.instrumented_methods[id(obj)]

            if func in funcs:
                old_path = funcs[func]

                if path != old_path:
                    logger.warning(
                        "Method %s was already instrumented on path %s. "
                        "Calls at %s may not be recorded.", func, old_path, path
                    )

                return

            else:

                funcs[func] = path

        else:
            funcs = dict()
            self.instrumented_methods[id(obj)] = funcs
            funcs[func] = path

    # WithInstrumentCallbacks requirement
    def get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, serial_utils.Lens]]:
        """
        Get the methods (rather the inner functions) matching the given `func`
        and the path of each.

        See [WithInstrumentCallbacks.get_methods_for_func][trulens_eval.instruments.WithInstrumentCallbacks.get_methods_for_func].
        """

        for _id, funcs in self.instrumented_methods.items():
            for f, path in funcs.items():
                if f == func:
                    yield (_id, f, path)

    # WithInstrumentCallbacks requirement
    def get_method_path(self, obj: object, func: Callable) -> serial_utils.Lens:
        """
        Get the path of the instrumented function `method` relative to this app.
        """

        # TODO: cleanup and/or figure out why references to objects change when executing langchain chains.

        assert isinstance(
            func, Callable
        ), f"Callable expected but got {python_utils.class_name(type(func))}."

        funcs = self.instrumented_methods.get(id(obj))

        if funcs is None:
            logger.warning(
                "A new object of type %s at %s is calling an instrumented method %s. "
                "The path of this call may be incorrect.",
                python_utils.class_name(type(obj)), python_utils.id_str(obj),
                python_utils.callable_name(func)
            )
            try:
                _id, _, path = next(iter(self.get_methods_for_func(func)))

            except Exception:
                logger.warning(
                    "No other objects use this function so cannot guess path."
                )
                return None

            logger.warning(
                "Guessing path of new object is %s based on other object (%s) using this function.",
                path, python_utils.id_str(_id)
            )

            funcs = {func: path}

            self.instrumented_methods[id(obj)] = funcs

            return path

        else:
            if func not in funcs:
                logger.warning(
                    "A new object of type %s at %s is calling an instrumented method %s. "
                    "The path of this call may be incorrect.",
                    python_utils.class_name(type(obj)),
                    python_utils.id_str(obj), python_utils.callable_name(func)
                )

                try:
                    _id, _, path = next(iter(self.get_methods_for_func(func)))
                except Exception:
                    logger.warning(
                        "No other objects use this function so cannot guess path."
                    )
                    return None

                logger.warning(
                    "Guessing path of new object is %s based on other object (%s) using this function.",
                    path, python_utils.id_str(_id)
                )

                return path

            else:

                return funcs.get(func)

    def json(self, *args, **kwargs):
        """Create a json string representation of this app."""
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json_utils.json_str_of_obj(
            self, *args, instrument=self.instrument, **kwargs
        )

    def model_dump(self, *args, redact_keys: bool = False, **kwargs):
        # Same problem as in json.
        return json_utils.jsonify(
            self,
            instrument=self.instrument,
            redact_keys=redact_keys,
            *args,
            **kwargs
        )

    # For use as a context manager.
    async def __aenter__(self):
        tracer: mod_trace.Tracer = mod_trace.get_tracer()
        # recording = tracer.__enter__()

        recording_span_ctx = await tracer.arecording()
        recording_span: mod_trace.PhantomSpanRecordingContext = await recording_span_ctx.__aenter__(
        )
        recording: mod_trace.RecordingContext = mod_trace.RecordingContext(
            app=self,
            tracer=tracer,
            span=recording_span,
            span_ctx=recording_span_ctx
        )
        recording_span.recording = recording
        recording_span.start_timestamp = time.time_ns()

        # recording.ctx = ctx

        token = self.recording_contexts.set(recording)
        recording.token = token

        return recording

    # For use as a context manager.
    def __enter__(self):
        tracer: mod_trace.Tracer = mod_trace.get_tracer()
        # recording = tracer.__enter__()

        recording_span_ctx = tracer.recording()
        recording_span: mod_trace.PhantomSpanRecordingContext = recording_span_ctx.__enter__(
        )
        recording: mod_trace.RecordingContext = mod_trace.RecordingContext(
            app=self,
            tracer=tracer,
            span=recording_span,
            span_ctx=recording_span_ctx
        )
        recording_span.recording = recording
        recording_span.start_timestamp = time.time_ns()

        # recording.ctx = ctx

        token = self.recording_contexts.set(recording)
        recording.token = token

        return recording

    # For use as a context manager.
    def __exit__(self, exc_type, exc_value, exc_tb):
        recording: mod_trace.RecordingContext = self.recording_contexts.get()

        assert recording is not None, "Not in a tracing context."
        assert recording.tracer is not None, "Not in a tracing context."

        recording.span.end_timestamp = time.time_ns()

        self.recording_contexts.reset(recording.token)
        return recording.span_ctx.__exit__(exc_type, exc_value, exc_tb)

    # For use as a context manager.
    async def __aexit__(self, exc_type, exc_value, exc_tb):
        recording: mod_trace.RecordingContext = self.recording_contexts.get()

        assert recording is not None, "Not in a tracing context."
        assert recording.tracer is not None, "Not in a tracing context."

        recording.span.end_timestamp = time.time_ns()

        self.recording_contexts.reset(recording.token)
        return await recording.span_ctx.__aexit__(exc_type, exc_value, exc_tb)

    # WithInstrumentCallbacks requirement
    def get_active_contexts(self) -> Iterable[mod_trace.RecordingContext]:
        """Get all active recording contexts."""

        recording = self.recording_contexts.get(contextvars.Token.MISSING)

        while recording is not contextvars.Token.MISSING:
            yield recording
            recording = recording.token.old_value

    # WithInstrumentCallbacks requirement
    def on_new_root_span(
        self,
        recording: mod_trace.RecordingContext,
        root_span: mod_trace.Span,
    ) -> record_schema.Record:

        tracer = root_span.context.tracer

        if self.tru.otel_exporter is not None:
            # Export to otel exporter if exporter was set in workspace.
            to_export = []
            for span in root_span.iter_family(include_phantom=True):
                e_span = span.otel_freeze()
                to_export.append(e_span)
                # print(e_span.name, "->", e_span.__class__.__name__)

            print(
                f"{text_utils.UNICODE_CHECK} Exporting {len(to_export)} spans to {self.tru.otel_exporter.__class__.__name__}."
            )
            self.tru.otel_exporter.export(to_export)

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

    def _check_instrumented(self, func):
        """
        Issue a warning and some instructions if a function that has not been
        instrumented is being used in a `with_` call.
        """

        if not isinstance(func, Callable):
            raise TypeError(
                f"Expected `func` to be a callable, but got {python_utils.class_name(type(func))}."
            )

        # If func is actually an object that implements __call__, check __call__
        # instead.
        if not (inspect.isfunction(func) or inspect.ismethod(func)):
            func = func.__call__

        if not python_utils.safe_hasattr(func, mod_trace.INSTRUMENT):
            if mod_trace.INSTRUMENT in dir(func):
                # HACK009: Need to figure out the __call__ accesses by class
                # name/object name with relation to this check for
                # instrumentation because we keep hitting spurious warnings
                # here. This is a temporary workaround.
                return

            logger.warning(
                """
Function %s has not been instrumented. This may be ok if it will call a function
that has been instrumented exactly once. Otherwise unexpected results may
follow. You can use `AddInstruments.method` of `trulens_eval.instruments` before
you use the `%s` wrapper to make sure `%s` does get instrumented. `%s` method
`print_instrumented` may be used to see methods that have been instrumented.
""", func, python_utils.class_name(self), python_utils.callable_name(func),
                python_utils.class_name(self)
            )

    async def awith_(
        self, func: asynchro_utils.CallableMaybeAwaitable[A, T], *args, **kwargs
    ) -> T:
        """
        Call the given async `func` with the given `*args` and `**kwargs` while
        recording, producing `func` results. The record of the computation is
        available through other means like the database or dashboard. If you
        need a record of this execution immediately, you can use `awith_record`
        or the `App` as a context mananger instead.
        """

        awaitable, _ = self.with_record(func, *args, **kwargs)

        if not isinstance(awaitable, Awaitable):
            raise TypeError(
                f"Expected `func` to be an async function or return an awaitable, but got {python_utils.class_name(type(awaitable))}."
            )

        return await awaitable

    async def with_(self, func: Callable[[A], T], *args, **kwargs) -> T:
        """
        Call the given async `func` with the given `*args` and `**kwargs` while
        recording, producing `func` results. The record of the computation is
        available through other means like the database or dashboard. If you
        need a record of this execution immediately, you can use `awith_record`
        or the `App` as a context mananger instead.
        """

        res, _ = self.with_record(func, *args, **kwargs)

        return res

    def with_record(
        self,
        func: Callable[[A], T],
        *args,
        record_metadata: JSON = None,
        **kwargs
    ) -> Tuple[T, record_schema.Record]:
        """
        Call the given `func` with the given `*args` and `**kwargs`, producing
        its results as well as a record of the execution.
        """

        self._check_instrumented(func)

        with self as ctx:
            ctx.record_metadata = record_metadata
            ret = func(*args, **kwargs)

        assert len(ctx.records) > 0, (
            f"Did not create any records. "
            f"This means that no instrumented methods were invoked in the process of calling {func}."
        )

        return ret, ctx.get()

    async def awith_record(
        self,
        func: Callable[[A], Awaitable[T]],
        *args,
        record_metadata: Optional[serial_utils.JSON] = None,
        **kwargs
    ) -> Tuple[T, record_schema.Record]:
        """
        Call the given `func` with the given `*args` and `**kwargs`, producing
        its results as well as a record of the execution.
        """

        awaitable, record = self.with_record(
            func, *args, record_metadata=record_metadata, **kwargs
        )
        if not isinstance(awaitable, Awaitable):
            raise TypeError(
                f"Expected `func` to be an async function or return an awaitable, but got {python_utils.class_name(type(awaitable))}."
            )

        return await awaitable, record

    def _throw_dep_message(
        self, method, is_async: bool = False, with_record: bool = False
    ):
        # Raises a deprecation message for the various methods that pass through to
        # wrapped app while recording.

        cname = self.__class__.__name__

        iscall = method == "__call__"

        old_method = f"""{method}{"_with_record" if with_record else ""}"""
        if iscall:
            old_method = f"""call{"_with_record" if with_record else ""}"""
        new_method = f"""{"a" if is_async else ""}with_{"record" if with_record else ""}"""

        app_callable = f"""app.{method}"""
        if iscall:
            app_callable = f"app"

        raise AttributeError(
            f"""
`{old_method}` is deprecated; To record results of your app's execution, use one of these options to invoke your app:
    (1) Use the `{"a" if is_async else ""}with_{"record" if with_record else ""}` method:
        ```python
        app # your app
        tru_app_recorder: {cname} = {cname}(app, ...)
        result{", record" if with_record else ""} = {"await " if is_async else ""}tru_app_recorder.{new_method}({app_callable}, ...args/kwargs-to-{app_callable}...)
        ```
    (2) Use {cname} as a context manager: 
        ```python
        app # your app
        tru_app_recorder: {cname} = {cname}(app, ...)
        with tru_app_recorder{" as records" if with_record else ""}:
            result = {"await " if is_async else ""}{app_callable}(...args/kwargs-to-{app_callable}...)
        {"record = records.get()" if with_record else ""}
        ```
"""
        )

    def _add_future_feedback(
        self, future_or_result: Union[
            feedback_schema.FeedbackResult,
            python_utils.Future[feedback_schema.FeedbackResult]]
    ) -> None:
        """
        Callback used to add feedback results to the database once they are
        done.
        
        See [_handle_record][trulens_eval.app.App._handle_record].
        """

        if isinstance(future_or_result, python_utils.Future):
            res = future_or_result.result()
        else:
            res = future_or_result

        self.tru.add_feedback(res)

    def _handle_record(
        self,
        record: record_schema.Record,
        feedback_mode: Optional[feedback_schema.FeedbackMode] = None
    ) -> Optional[List[Tuple[mod_feedback.Feedback, python_utils.
                             Future[feedback_schema.FeedbackResult]]]]:
        """
        Write out record-related info to database if set and schedule feedback
        functions to be evaluated. If feedback_mode is provided, will use that
        mode instead of the one provided to constructor.
        """

        if feedback_mode is None:
            feedback_mode = self.feedback_mode

        if self.tru is None or self.feedback_mode is None:
            return None

        self.tru: Tru
        self.db: DB

        # Need to add record to db before evaluating feedback functions.
        record_id = self.tru.add_record(record=record)

        if len(self.feedbacks) == 0:
            return []

        # Add empty (to run) feedback to db.
        if feedback_mode == feedback_schema.FeedbackMode.DEFERRED:
            for f in self.feedbacks:
                self.db.insert_feedback(
                    feedback_schema.FeedbackResult(
                        name=f.name,
                        record_id=record_id,
                        feedback_definition_id=f.feedback_definition_id
                    )
                )

            return None

        elif feedback_mode in [feedback_schema.FeedbackMode.WITH_APP,
                               feedback_schema.FeedbackMode.WITH_APP_THREAD]:

            return self.tru._submit_feedback_functions(
                record=record,
                feedback_functions=self.feedbacks,
                app=self,
                on_done=self._add_future_feedback
            )

    def _handle_error(self, record: record_schema.Record, error: Exception):
        if self.db is None:
            return

    def __getattr__(self, __name: str) -> Any:
        # A message for cases where a user calls something that the wrapped app
        # contains. We do not support this form of pass-through calls anymore.

        if python_utils.safe_hasattr(self.app, __name):
            msg = ATTRIBUTE_ERROR_MESSAGE.format(
                attribute_name=__name,
                class_name=type(self).__name__,
                app_class_name=type(self.app).__name__
            )
            raise AttributeError(msg)

        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{__name}'"
            )

    def dummy_record(
        self,
        cost: base_schema.Cost = base_schema.Cost(),
        perf: base_schema.Perf = base_schema.Perf.now(),
        ts: datetime.datetime = datetime.datetime.now(),
        main_input: str = "main_input are strings.",
        main_output: str = "main_output are strings.",
        main_error: str = "main_error are strings.",
        meta: Dict = {'metakey': 'meta are dicts'},
        tags: str = 'tags are strings'
    ) -> record_schema.Record:
        """Create a dummy record with some of the expected structure without
        actually invoking the app.

        The record is a guess of what an actual record might look like but will
        be missing information that can only be determined after a call is made.

        All args are [Record][trulens_eval.schema.record.Record] fields except these:

            - `record_id` is generated using the default id naming schema.
            - `app_id` is taken from this recorder.
            - `calls` field is constructed based on instrumented methods. 
        """

        calls = []

        for methods in self.instrumented_methods.values():
            for func, lens in methods.items():

                component = lens.get_sole_item(self)

                if not hasattr(component, func.__name__):
                    continue
                method = getattr(component, func.__name__)

                sig = inspect.signature(method)

                method_serial = pyschema_utils.FunctionOrMethod.of_callable(
                    method
                )

                sample_args = {}
                for p in sig.parameters.values():
                    if p.default == inspect.Parameter.empty:
                        sample_args[p.name] = None
                    else:
                        sample_args[p.name] = p.default

                sample_call = record_schema.RecordAppCall(
                    stack=[
                        record_schema.RecordAppCallMethod(
                            path=lens, method=method_serial
                        )
                    ],
                    args=sample_args,
                    rets=None,
                    pid=0,
                    tid=0
                )

                calls.append(sample_call)

        return record_schema.Record(
            app_id=self.app_id,
            calls=calls,
            cost=cost,
            perf=perf,
            ts=ts,
            main_input=main_input,
            main_output=main_output,
            main_error=main_error,
            meta=meta,
            tags=tags
        )

    def instrumented(
        self
    ) -> Iterable[Tuple[serial_utils.Lens, serial_utils.JSON]]:
        """
        Iteration over instrumented components and their categories.
        """

        for q, c in instrumented_component_views(self.model_dump()):
            # Add the chain indicator so the resulting paths can be specified
            # for feedback selectors.
            q = serial_utils.Lens(
                path=(
                    serial_utils.
                    GetItemOrAttribute(item_or_attribute="__app__"),
                ) + q.path
            )
            yield q, c

    def print_instrumented(self) -> None:
        """Print the instrumented components and methods."""

        print("Components:")
        self.print_instrumented_components()
        print("\nMethods:")
        self.print_instrumented_methods()

    def format_instrumented_methods(self) -> str:
        """Build a string containing a listing of instrumented methods."""

        return "\n".join(
            f"Object at 0x{obj:x}:\n\t" + "\n\t".join(
                f"{m} with path {feedback_schema.Select.App + path}"
                for m, path in p.items()
            )
            for obj, p in self.instrumented_methods.items()
        )

    def print_instrumented_methods(self) -> None:
        """Print instrumented methods."""

        print(self.format_instrumented_methods())

    def print_instrumented_components(self) -> None:
        """Print instrumented components and their categories."""

        object_strings = []

        for t in self.instrumented():
            t: Tuple[serial_utils.Lens, serial_utils.JSON]
            path = serial_utils.Lens(t[0].path[1:])
            obj = next(iter(path.get(self)))
            object_strings.append(
                f"\t{type(obj).__name__} ({obj.__class__.__name__}) at 0x{id(obj):x} with path {str(t[0])}"
            )

        print("\n".join(object_strings))


# NOTE: Cannot App.model_rebuild here due to circular imports involving tru.Tru
# and database.base.DB. Will rebuild each App subclass instead.

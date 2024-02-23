"""
Generalized root type for various libraries like llama_index and langchain .
"""

from abc import ABC
from abc import abstractmethod
import contextvars
import inspect
from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
import queue
import threading
from threading import Lock
from typing import (
    Any, Awaitable, Callable, ClassVar, Dict, Hashable, Iterable, List,
    Optional, Sequence, Set, Tuple, Type, TypeVar
)

import pydantic

from trulens_eval.db import DB
from trulens_eval.feedback import Feedback
from trulens_eval.instruments import Instrument
from trulens_eval.instruments import WithInstrumentCallbacks
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import Perf
from trulens_eval.schema import Record
from trulens_eval.schema import RecordAppCall
from trulens_eval.schema import Select
from trulens_eval.tru import Tru
from trulens_eval.utils.asynchro import CallableMaybeAwaitable
from trulens_eval.utils.asynchro import desync
from trulens_eval.utils.asynchro import sync
from trulens_eval.utils.json import json_str_of_obj
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import CLASS_INFO
from trulens_eval.utils.python import callable_name
from trulens_eval.utils.python import class_name
from trulens_eval.utils.python import \
    Future  # can take type args with python < 3.9
from trulens_eval.utils.python import id_str
from trulens_eval.utils.python import \
    Queue  # can take type args with python < 3.9
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import T
from trulens_eval.utils.serial import all_objects
from trulens_eval.utils.serial import GetItemOrAttribute
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import JSON_BASES
from trulens_eval.utils.serial import JSON_BASES_T
from trulens_eval.utils.serial import Lens

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# App component.
COMPONENT = Any

A = TypeVar("A")

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


class ComponentView(ABC):
    """
    Views of common app component types for sorting them and displaying them in
    some unified manner in the UI. Operates on components serialized into json
    dicts representing various components, not the components themselves.
    """

    def __init__(self, json: JSON):
        self.json = json
        self.cls = Class.of_class_info(json)

    @staticmethod
    def of_json(json: JSON) -> 'ComponentView':
        """
        Sort the given json into the appropriate component view type.
        """

        cls = Class.of_class_info(json)

        if LangChainComponent.class_is(cls):
            return LangChainComponent.of_json(json)
        elif LlamaIndexComponent.class_is(cls):
            return LlamaIndexComponent.of_json(json)
        elif TrulensComponent.class_is(cls):
            return TrulensComponent.of_json(json)
        elif CustomComponent.class_is(cls):
            return CustomComponent.of_json(json)
        else:
            # TODO: custom class

            raise TypeError(f"Unhandled component type with class {cls}")

    @staticmethod
    @abstractmethod
    def class_is(cls: Class) -> bool:
        """
        Determine whether the given class representation `cls` is of the type to
        be viewed as this component type.
        """
        pass

    def unsorted_parameters(self, skip: Set[str]) -> Dict[str, JSON_BASES_T]:
        """
        All basic parameters not organized by other accessors.
        """

        ret = {}

        for k, v in self.json.items():
            if k not in skip and isinstance(v, JSON_BASES):
                ret[k] = v

        return ret

    @staticmethod
    def innermost_base(
        bases: Optional[Sequence[Class]] = None,
        among_modules=set(["langchain", "llama_index", "trulens_eval"])
    ) -> Optional[str]:
        """
        Given a sequence of classes, return the first one which comes from one
        of the `among_modules`. You can use this to determine where ultimately
        the encoded class comes from in terms of langchain, llama_index, or
        trulens_eval even in cases they extend each other's classes. Returns
        None if no module from `among_modules` is named in `bases`.
        """
        if bases is None:
            return None

        for base in bases:
            if "." in base.module.module_name:
                root_module = base.module.module_name.split(".")[0]
            else:
                root_module = base.module.module_name

            if root_module in among_modules:
                return root_module

        return None


class LangChainComponent(ComponentView):

    @staticmethod
    def class_is(cls: Class) -> bool:
        if ComponentView.innermost_base(cls.bases) == "langchain":
            return True

        return False

    @staticmethod
    def of_json(json: JSON) -> 'LangChainComponent':
        from trulens_eval.utils.langchain import component_of_json
        return component_of_json(json)


class LlamaIndexComponent(ComponentView):

    @staticmethod
    def class_is(cls: Class) -> bool:
        if ComponentView.innermost_base(cls.bases) == "llama_index":
            return True

        return False

    @staticmethod
    def of_json(json: JSON) -> 'LlamaIndexComponent':
        from trulens_eval.utils.llama import component_of_json
        return component_of_json(json)


class TrulensComponent(ComponentView):
    """
    Components provided in trulens.
    """

    @staticmethod
    def class_is(cls: Class) -> bool:
        if ComponentView.innermost_base(cls.bases) == "trulens_eval":
            return True

        #if any(base.module.module_name.startswith("trulens.") for base in cls.bases):
        #    return True

        return False

    @staticmethod
    def of_json(json: JSON) -> 'TrulensComponent':
        from trulens_eval.utils.trulens import component_of_json
        return component_of_json(json)


class Prompt(ComponentView):
    # langchain.prompts.base.BasePromptTemplate
    # llama_index.prompts.base.Prompt

    @property
    @abstractmethod
    def template(self) -> str:
        pass


class LLM(ComponentView):
    # langchain.llms.base.BaseLLM
    # llama_index.llms.base.LLM

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class Tool(ComponentView):
    # langchain ???
    # llama_index.tools.types.BaseTool

    @property
    @abstractmethod
    def tool_name(self) -> str:
        pass


class Agent(ComponentView):
    # langchain ???
    # llama_index.agent.types.BaseAgent

    @property
    @abstractmethod
    def agent_name(self) -> str:
        pass


class Memory(ComponentView):
    # langchain.schema.BaseMemory
    # llama_index ???
    pass


class Other(ComponentView):
    # Any component that does not fit into the other named categories.
    pass


class CustomComponent(ComponentView):

    class Custom(Other):
        # No categorization of custom class components for now. Using just one
        # "Custom" catch-all.

        @staticmethod
        def class_is(cls: Class) -> bool:
            return True

    COMPONENT_VIEWS = [Custom]

    @staticmethod
    def constructor_of_class(cls: Class) -> Type['CustomComponent']:
        for view in CustomComponent.COMPONENT_VIEWS:
            if view.class_is(cls):
                return view

        raise TypeError(f"Unknown custom component type with class {cls}")

    @staticmethod
    def component_of_json(json: JSON) -> 'CustomComponent':
        cls = Class.of_class_info(json)

        view = CustomComponent.constructor_of_class(cls)

        return view(json)

    @staticmethod
    def class_is(cls: Class) -> bool:
        # Assumes this is the last check done.
        return True

    @staticmethod
    def of_json(json: JSON) -> 'CustomComponent':
        return CustomComponent.component_of_json(json)


def instrumented_component_views(
    obj: object
) -> Iterable[Tuple[Lens, ComponentView]]:
    """
    Iterate over contents of `obj` that are annotated with the CLASS_INFO
    attribute/key. Returns triples with the accessor/selector, the Class object
    instantiated from CLASS_INFO, and the annotated object itself.
    """

    for q, o in all_objects(obj):
        if isinstance(o, pydantic.BaseModel) and CLASS_INFO in o.model_fields:
            yield q, ComponentView.of_json(json=o)

        if isinstance(o, Dict) and CLASS_INFO in o:
            yield q, ComponentView.of_json(json=o)


class RecordingContext():
    """
    Manager of the creation of records from record calls. Each instance of this
    class will result in a record for every "root" instrumented method called.
    Root method here means the first instrumented method in a call stack. Note
    that there may be more than one of these contexts in play at the same time
    due to:

    - More than one wrapper of the same app.
    - More than one context manager ("with" statement) surrounding calls to the
      same app.
    - Calls to "with_record" on methods that themselves contain recording.
    - Calls to apps that use trulens internally to track records in any of the
      supported ways.
    - Combinations of the above.
    """

    def __init__(self, app: 'App', record_metadata: JSON = None):
        # A record (in terms of its RecordAppCall) in process of being created
        # are kept here:
        self.calls: List[RecordAppCall] = []

        # Completed records go here:
        self.records: List[Record] = []

        # Lock calls and records when adding calls or finishing a record.
        self.lock: Lock = Lock()

        # Token for context management. See contextvars.
        self.token: contextvars.Token = None

        # App managing this recording.
        self.app: WithInstrumentCallbacks = app

        # Metadata to attach to all records produced in this context.
        self.record_metadata = record_metadata

    def __iter__(self):
        return iter(self.records)

    def get(self) -> Record:
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

    def __getitem__(self, idx: int) -> Record:
        return self.records[idx]

    def __len__(self):
        return len(self.records)

    def __hash__(self) -> int:
        # The same app can have multiple recording contexts.
        return hash(id(self.app)) + hash(id(self.records))

    def __eq__(self, other):
        return hash(self) == hash(other)
        # return id(self.app) == id(other.app) and id(self.records) == id(other.records)

    def add_call(self, call: RecordAppCall):
        """
        Add the given call to the currently tracked call list.
        """
        with self.lock:
            self.calls.append(call)

    def finish_record(
        self,
        calls_to_record: Callable[[List[RecordAppCall], JSON], Record],
        existing_record: Optional[Record] = None
    ):
        """
        Run the given function to build a record from the tracked calls and any
        pre-specified metadata.
        """

        with self.lock:
            record = calls_to_record(
                self.calls,
                self.record_metadata,
                existing_record=existing_record
            )
            self.calls = []

            if existing_record is None:
                # If existing record was given, we assume it was already
                # inserted into this list.
                self.records.append(record)

        return record


class App(AppDefinition, WithInstrumentCallbacks, Hashable):
    """
    Generalization of a wrapped model.
    
    Non-serialized fields here while the serialized ones are defined in
    [AppDefinition][trulens_eval.schema.AppDefinition].
    """

    model_config: ClassVar[dict] = {
        # Tru, DB, most of the types on the excluded fields.
        'arbitrary_types_allowed': True
    }

    feedbacks: List[Feedback] = pydantic.Field(
        exclude=True, default_factory=list
    )
    """Feedback functions to evaluate on each record."""

    tru: Optional[Tru] = pydantic.Field(default=None, exclude=True)
    """Workspace manager."""

    db: Optional[DB] = pydantic.Field(default=None, exclude=True)
    """Database interfaces."""

    app: Any = pydantic.Field(exclude=True)
    """The app to be recorded."""

    instrument: Optional[Instrument] = pydantic.Field(None, exclude=True)
    """Instrumentation class.
    
    This is needed for serialization as it tells us which objects we want to be
    included in the json representation of this app.
    """

    recording_contexts: contextvars.ContextVar[RecordingContext] \
        = pydantic.Field(None, exclude=True)
    """Sequnces of records produced by the this class used as a context manager
    are stored in a RecordingContext.
    
    Using a context var so that context managers can be nested.
    """

    instrumented_methods: Dict[int, Dict[Callable, Lens]] = \
        pydantic.Field(exclude=True, default_factory=dict)
    """Mapping of instrumented methods (by id(.) of owner object and the
    function) to their path in this app."""

    records_with_pending_feedback_results: Queue[Record] = \
        pydantic.Field(exclude=True, default_factory=lambda: queue.Queue(maxsize=1024))
    """EXPRIMENTAL: Records produced by this app which might have yet to finish
    feedback runs."""

    manage_pending_feedback_results_thread: Optional[threading.Thread] = \
        pydantic.Field(exclude=True, default=None)
    """Thread for manager of pending feedback results queue. See
    _manage_pending_feedback_results."""

    def __init__(
        self,
        tru: Optional[Tru] = None,
        feedbacks: Optional[Iterable[Feedback]] = None,
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
                obj=self.app, query=Select.Query().app
            )
        else:
            pass

        if self.feedback_mode == FeedbackMode.WITH_APP_THREAD:
            # EXPERIMENTAL: Start the thread that manages the queue of records
            # with pending feedback results. This is meant to be run
            # permentantly in a separate thread. It will remove records from the
            # queue `records_with_pending_feedback_results` as their feedback
            # results are computed and makes sure the queue does not keep
            # growing.
            self._start_manage_pending_feedback_results()

        self._tru_post_init()

    def __del__(self):
        # Can use to do things when this object is being garbage collected.
        pass

    def _start_manage_pending_feedback_results(self) -> None:
        """
        EXPERIMENTAL: Start the thread that manages the queue of records with
        pending feedback results. This is meant to be run permentantly in a
        separate thread. It will remove records from the queue
        `records_with_pending_feedback_results` as their feedback results are
        computed and makes sure the queue does not keep growing.
        """

        if self.manage_pending_feedback_results_thread is not None:
            raise RuntimeError("Manager Thread already started.")

        self.manage_pending_feedback_results_thread = threading.Thread(
            target=self._manage_pending_feedback_results,
            daemon=True  # otherwise this thread will keep parent alive
        )
        self.manage_pending_feedback_results_thread.start()

    def _manage_pending_feedback_results(self) -> None:
        """
        EXPERIMENTAL: Manage the queue of records with pending feedback results.
        This is meant to be run permentantly in a separate thread. It will
        remove records from the queue records_with_pending_feedback_results as
        their feedback results are computed and makes sure the queue does not
        keep growing.
        """

        while True:
            record = self.records_with_pending_feedback_results.get()
            record.wait_for_feedback_results()

    def wait_for_feedback_results(self) -> None:
        """Wait for all feedbacks functions to complete.
         
        This applies to all feedbacks on all records produced by this app. This
        call will block until finished and if new records are produced while
        this is running, it will include them.
        """

        while not self.records_with_pending_feedback_results.empty():
            record = self.records_with_pending_feedback_results.get()

            record.wait_for_feedback_results()

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

        raise ValueError(
            f"Could not determine context from unrecognized `app` type {type(app)}."
        )

    def __hash__(self):
        return hash(id(self))

    def _tru_post_init(self):
        """
        Database-related initialization.
        """

        if self.tru is None:
            if self.feedback_mode != FeedbackMode.NONE:
                logger.debug("Creating default tru.")
                self.tru = Tru()

        else:
            if self.feedback_mode == FeedbackMode.NONE:
                logger.warning(
                    "`tru` is specified but `feedback_mode` is FeedbackMode.NONE. "
                    "No feedback evaluation and logging will occur."
                )

        if self.tru is not None:
            self.db = self.tru.db

            self.db.insert_app(app=self)

            if self.feedback_mode != FeedbackMode.NONE:
                logger.debug("Inserting feedback function definitions to db.")

                for f in self.feedbacks:
                    self.db.insert_feedback_definition(f)

        else:
            if len(self.feedbacks) > 0:
                raise ValueError(
                    "Feedback logging requires `tru` to be specified."
                )

        if self.feedback_mode == FeedbackMode.DEFERRED:
            for f in self.feedbacks:
                # Try to load each of the feedback implementations. Deferred
                # mode will do this but we want to fail earlier at app
                # constructor here.
                try:
                    f.implementation.load()
                except Exception as e:
                    raise Exception(
                        f"Feedback function {f} is not loadable. Cannot use DEFERRED feedback mode. {e}"
                    )

    def main_call(self, human: str) -> str:
        """If available, a single text to a single text invocation of this app."""

        if self.__class__.main_acall is not App.main_acall:
            # Use the async version if available.
            return sync(self.main_acall, human)

        raise NotImplementedError()

    async def main_acall(self, human: str) -> str:
        """If available, a single text to a single text invocation of this app."""

        if self.__class__.main_call is not App.main_call:
            logger.warning("Using synchronous version of main call.")
            # Use the sync version if available.
            return await desync(self.main_call, human)

        raise NotImplementedError()

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
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

        while not isinstance(focus, JSON_BASES) and len(focus) == 1:
            focus = focus[0]

            if isinstance(focus, pydantic.BaseModel):
                focus = list(focus.model_dump().values())
                continue

            if isinstance(focus, Dict):
                focus = list(focus.values())
                continue

            if not isinstance(focus, Sequence):
                break

        if isinstance(focus, JSON_BASES):
            return str(focus)

        # Otherwise we are not sure.
        logger.warning(
            "Unsure what the main input string is for the call to %s with args %s.",
            callable_name(func), all_args
        )

        # After warning, just take the first item in each container until a
        # non-container is reached.
        focus = all_args
        while not isinstance(focus, JSON_BASES) and len(focus) >= 1:
            focus = focus[0]

            if isinstance(focus, pydantic.BaseModel):
                focus = list(focus.model_dump().values())
                continue

            if isinstance(focus, Dict):
                focus = list(focus.values())
                continue

            if not isinstance(focus, Sequence):
                break

        if isinstance(focus, JSON_BASES):
            return str(focus)

        logger.warning(
            "Could not determine main input string call to %s with args %s.",
            callable_name(func), all_args
        )

        return None

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> str:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        if isinstance(ret, str):
            return ret

        if isinstance(ret, float):
            return str(ret)

        if isinstance(ret, Dict):
            return next(iter(ret.values()))

        elif isinstance(ret, Sequence):
            if len(ret) > 0:
                return str(ret[0])
            else:
                return None

        else:
            logger.warning(
                f"Unsure what the main output string is for the call to {callable_name(func)} with return type {type(ret)}."
            )
            return str(ret)

    # WithInstrumentCallbacks requirement
    def on_method_instrumented(self, obj: object, func: Callable, path: Lens):
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
                        f"Method {func} was already instrumented on path {old_path}. "
                        f"Calls at {path} may not be recorded."
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
    ) -> Iterable[Tuple[int, Callable, Lens]]:
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
    def get_method_path(self, obj: object, func: Callable) -> Lens:
        """
        Get the path of the instrumented function `method` relative to this app.
        """

        # TODO: cleanup and/or figure out why references to objects change when executing langchain chains.

        funcs = self.instrumented_methods.get(id(obj))

        if funcs is None:
            logger.warning(
                "A new object of type %s at %s is calling an instrumented method %s. "
                "The path of this call may be incorrect.",
                class_name(type(obj)), id_str(obj), callable_name(func)
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
                path, id_str(_id)
            )

            funcs = {func: path}

            self.instrumented_methods[id(obj)] = funcs

            return path

        else:
            if func not in funcs:
                logger.warning(
                    "A new object of type %s at %s is calling an instrumented method %s. "
                    "The path of this call may be incorrect.",
                    class_name(type(obj)), id_str(obj), callable_name(func)
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
                    path, id_str(_id)
                )

                return path

            else:

                return funcs.get(func)

    def json(self, *args, **kwargs):
        """Create a json string representation of this app."""
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json_str_of_obj(
            self, *args, instrument=self.instrument, **kwargs
        )

    def model_dump(self, *args, redact_keys: bool = False, **kwargs):
        # Same problem as in json.
        return jsonify(
            self,
            instrument=self.instrument,
            redact_keys=redact_keys,
            *args,
            **kwargs
        )

    # For use as a context manager.
    def __enter__(self):
        ctx = RecordingContext(app=self)

        token = self.recording_contexts.set(ctx)
        ctx.token = token

        return ctx

    # For use as a context manager.
    def __exit__(self, exc_type, exc_value, exc_tb):
        ctx = self.recording_contexts.get()
        self.recording_contexts.reset(ctx.token)

        if exc_type is not None:
            raise exc_value

        return

    # WithInstrumentCallbacks requirement
    def on_new_record(self, func) -> Iterable[RecordingContext]:
        """Called at the start of record creation.

        See
        [WithInstrumentCallbacks.on_new_record][trulens_eval.instruments.WithInstrumentCallbacks.on_new_record].
        """
        ctx = self.recording_contexts.get(contextvars.Token.MISSING)

        while ctx is not contextvars.Token.MISSING:
            yield ctx
            ctx = ctx.token.old_value

    # WithInstrumentCallbacks requirement
    def on_add_record(
        self,
        ctx: RecordingContext,
        func: Callable,
        sig: Signature,
        bindings: BoundArguments,
        ret: Any,
        error: Any,
        perf: Perf,
        cost: Cost,
        existing_record: Optional[Record] = None
    ) -> Record:
        """Called by instrumented methods if they use _new_record to construct a record call list.

        See [WithInstrumentCallbacks.on_add_record][trulens_eval.instruments.WithInstrumentCallbacks.on_add_record].
        """

        def build_record(
            calls: Iterable[RecordAppCall],
            record_metadata: JSON,
            existing_record: Optional[Record] = None
        ) -> Record:
            calls = list(calls)

            assert len(calls) > 0, "No information recorded in call."

            main_in = self.main_input(func, sig, bindings)
            main_out = self.main_output(func, sig, bindings, ret)

            updates = dict(
                main_input=jsonify(main_in),
                main_output=jsonify(main_out),
                main_error=jsonify(error),
                calls=calls,
                cost=cost,
                perf=perf,
                app_id=self.app_id,
                tags=self.tags,
                meta=jsonify(record_metadata)
            )

            if existing_record is not None:
                existing_record.update(**updates)
            else:
                existing_record = Record(**updates)

            return existing_record

        # Finishing record needs to be done in a thread lock, done there:
        record = ctx.finish_record(
            build_record, existing_record=existing_record
        )

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

        if self.feedback_mode == FeedbackMode.WITH_APP_THREAD:
            # Add the record to ones with pending feedback.

            self.records_with_pending_feedback_results.put(record)

        elif self.feedback_mode == FeedbackMode.WITH_APP:
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
                f"Expected `func` to be a callable, but got {class_name(type(func))}."
            )

        # If func is actually an object that implements __call__, check __call__
        # instead.
        if not (inspect.isfunction(func) or inspect.ismethod(func)):
            func = func.__call__

        if not safe_hasattr(func, Instrument.INSTRUMENT):
            if Instrument.INSTRUMENT in dir(func):
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
""", func, class_name(self), callable_name(func), class_name(self)
            )

    async def awith_(
        self, func: CallableMaybeAwaitable[A, T], *args, **kwargs
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
                f"Expected `func` to be an async function or return an awaitable, but got {class_name(type(awaitable))}."
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
    ) -> Tuple[T, Record]:
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
        record_metadata: JSON = None,
        **kwargs
    ) -> Tuple[T, Record]:
        """
        Call the given `func` with the given `*args` and `**kwargs`, producing
        its results as well as a record of the execution.
        """

        awaitable, record = self.with_record(
            func, *args, record_metadata=record_metadata, **kwargs
        )
        if not isinstance(awaitable, Awaitable):
            raise TypeError(
                f"Expected `func` to be an async function or return an awaitable, but got {class_name(type(awaitable))}."
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

    def _add_future_feedback(self, future_result: Future[FeedbackResult]):
        """
        Callback used to add feedback results to the database once they are
        done. See `App._handle_record`.
        """
        res = future_result.result()
        self.tru.add_feedback(res)

    def _handle_record(
        self,
        record: Record,
        feedback_mode: Optional[FeedbackMode] = None
    ) -> Optional[List[Tuple[Feedback, Future[FeedbackResult]]]]:
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
        if feedback_mode == FeedbackMode.DEFERRED:
            for f in self.feedbacks:
                self.db.insert_feedback(
                    FeedbackResult(
                        name=f.name,
                        record_id=record_id,
                        feedback_definition_id=f.feedback_definition_id
                    )
                )

            return None

        elif feedback_mode in [FeedbackMode.WITH_APP,
                               FeedbackMode.WITH_APP_THREAD]:

            return self.tru._submit_feedback_functions(
                record=record,
                feedback_functions=self.feedbacks,
                app=self,
                on_done=self._add_future_feedback
            )

    def _handle_error(self, record: Record, error: Exception):
        if self.db is None:
            return

    def __getattr__(self, __name: str) -> Any:
        # A message for cases where a user calls something that the wrapped app
        # contains. We do not support this form of pass-through calls anymore.

        if safe_hasattr(self.app, __name):
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

    def instrumented(self) -> Iterable[Tuple[Lens, ComponentView]]:
        """
        Enumerate instrumented components and their categories.
        """

        for q, c in instrumented_component_views(self.model_dump()):
            # Add the chain indicator so the resulting paths can be specified
            # for feedback selectors.
            q = Lens(
                path=(GetItemOrAttribute(item_or_attribute="__app__"),) + q.path
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
            f"Object at 0x{obj:x}:\n\t" + "\n\t".
            join(f"{m} with path {Select.App + path}"
                 for m, path in p.items())
            for obj, p in self.instrumented_methods.items()
        )

    def print_instrumented_methods(self) -> None:
        """
        Print instrumented methods.
        """

        print(self.format_instrumented_methods())

    def print_instrumented_components(self) -> None:
        """Print instrumented components and their categories."""

        object_strings = []

        for t in self.instrumented():
            path = Lens(t[0].path[1:])
            obj = next(iter(path.get(self)))
            object_strings.append(
                f"\t{type(obj).__name__} ({t[1].__class__.__name__}) at 0x{id(obj):x} with path {str(t[0])}"
            )

        print("\n".join(object_strings))

"""
Generalized root type for various libraries like llama_index and langchain .
"""

from abc import ABC
from abc import abstractmethod
from concurrent import futures
from concurrent.futures import as_completed
import contextvars
import inspect
from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from threading import Lock
from typing import (
    Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Set,
    Tuple, Type
)

import dill
import pydantic
from pydantic import Field

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
from trulens_eval.utils.json import json_str_of_obj
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import callable_name
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import CLASS_INFO
from trulens_eval.utils.pyschema import ObjSerial
from trulens_eval.utils.serial import all_objects
from trulens_eval.utils.serial import GetItemOrAttribute
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import JSON_BASES
from trulens_eval.utils.serial import JSON_BASES_T
from trulens_eval.utils.serial import JSONPath
from trulens_eval.utils.serial import SerialModel
from trulens_eval.utils.threading import TP

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# App component.
COMPONENT = Any


class ComponentView(ABC):
    """
    Views of common app component types for sorting them and displaying them in
    some unified manner in the UI. Operates on components serialized into json
    dicts representing various components, not the components themselves.
    """

    def __init__(self, json: JSON):
        self.json = json
        self.cls = Class.of_json(json)

    @staticmethod
    def of_json(json: JSON) -> 'ComponentView':
        """
        Sort the given json into the appropriate component view type.
        """

        cls = Class.of_json(json)

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
        bases: Sequence[Class],
        among_modules=set(["langchain", "llama_index", "trulens_eval"])
    ) -> str:
        """
        Given a sequence of classes, return the first one which comes from one
        of the `among_modules`. You can use this to determine where ultimately
        the encoded class comes from in terms of langchain, llama_index, or
        trulens_eval even in cases they extend each other's classes. Returns
        None if no module from `among_modules` is named in `bases`.
        """

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
        cls = Class.of_json(json)

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
) -> Iterable[Tuple[JSONPath, ComponentView]]:
    """
    Iterate over contents of `obj` that are annotated with the CLASS_INFO
    attribute/key. Returns triples with the accessor/selector, the Class object
    instantiated from CLASS_INFO, and the annotated object itself.
    """

    for q, o in all_objects(obj):
        if isinstance(o, pydantic.BaseModel) and CLASS_INFO in o.__fields__:
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
        self, calls_to_record: Callable[[List[RecordAppCall], JSON], Record]
    ):
        """
        Run the given function to build a record from the tracked calls and any
        pre-specified metadata.
        """

        with self.lock:
            record = calls_to_record(
                self.calls, record_metadata=self.record_metadata
            )
            self.calls = []
            self.records.append(record)

        return record


class App(AppDefinition, SerialModel, WithInstrumentCallbacks, Hashable):
    """
    Generalization of a wrapped model.
    """

    # Non-serialized fields here while the serialized ones are defined in
    # `schema.py:App`.

    # Feedback functions to evaluate on each record.
    feedbacks: Sequence[Feedback] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    # NOTE: Maybe move to schema.App .
    tru: Optional[Tru] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    # NOTE: Maybe mobe to schema.App .
    db: Optional[DB] = Field(exclude=True)

    # The wrapped app.
    app: Any = Field(exclude=True)

    # Instrumentation class.
    instrument: Instrument = Field(exclude=True)

    # Sequnces of records produced by the this class used as a context manager.
    # Using a context var so that context managers can be nested.
    recording_contexts: contextvars.ContextVar[Sequence[RecordingContext]
                                              ] = Field(exclude=True)

    # Mapping of instrumented methods (by id(.) of owner object and the
    # function) to their path in this app:
    instrumented_methods: Dict[int, Dict[Callable, JSONPath]] = Field(
        exclude=True, default_factory=dict
    )

    def __init__(
        self,
        tru: Optional[Tru] = None,
        feedbacks: Optional[Sequence[Feedback]] = None,
        **kwargs
    ):

        feedbacks = feedbacks or []

        # for us:
        kwargs['tru'] = tru
        kwargs['feedbacks'] = feedbacks
        kwargs['recording_contexts'] = contextvars.ContextVar(
            "recording_contexts"
        )

        # Cannot use this to set app. AppDefinition has app as JSON type.
        # TODO: Figure out a better design to avoid this.
        super().__init__(**kwargs)

        app = kwargs['app']
        self.app = app

        self.instrument.instrument_object(
            obj=self.app, query=Select.Query().app
        )

    def __hash__(self):
        return hash(id(self))

    def post_init(self):
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
                f.implementation.load()

    def main_call(self, human: str) -> str:
        # If available, a single text to a single text invocation of this app.
        raise NotImplementedError()

    async def main_acall(self, human: str) -> str:
        # If available, a single text to a single text invocation of this app.
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
        if len(all_args) == 1 and isinstance(all_args[0], str):
            return all_args[0]

        # Otherwise we are not sure.
        logger.warning(
            f"Unsure what the main input string is for the call to {callable_name(func)} with args {all_args}."
        )

        if len(all_args) > 0:
            return all_args[0]
        else:
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
                return ret[0]
            else:
                return None

        else:
            logger.warning(
                f"Unsure what the main output string is for the call to {callable_name(func)}."
            )
            return str(ret)

    # WithInstrumentCallbacks requirement
    def _on_method_instrumented(
        self, obj: object, func: Callable, path: JSONPath
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
    def _get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, JSONPath]]:
        """
        Get the methods (rather the inner functions) matching the given `func`
        and the path of each.
        """

        for _id, funcs in self.instrumented_methods.items():
            for f, path in funcs.items():
                """
                # TODO: wider wrapping support
                if hasattr(f, "__func__"):
                    if method.__func__ == func:
                        yield (method, path) 
                else:
                """
                if f == func:
                    yield (_id, f, path)

    # WithInstrumentCallbacks requirement
    def _get_method_path(self, obj: object, func: Callable) -> JSONPath:
        """
        Get the path of the instrumented function `method` relative to this app.
        """

        # TODO: cleanup and/or figure out why references to objects change when executing langchain chains.

        funcs = self.instrumented_methods.get(id(obj))

        if funcs is None:
            logger.warning(
                f"A new object of type {type(obj)} at 0x{id(obj):x} is calling an instrumented method {func}. "
                "The path of this call may be incorrect."
            )
            try:
                _id, f, path = next(iter(self._get_methods_for_func(func)))
            except Exception:
                logger.warning(
                    "No other objects use this function so cannot guess path."
                )
                return None

            logger.warning(
                f"Guessing path of new object is {path} based on other object (0x{_id:x}) using this function."
            )

            funcs = {func: path}

            self.instrumented_methods[id(obj)] = funcs

            return path

        else:
            if func not in funcs:
                logger.warning(
                    f"A new object of type {type(obj)} at 0x{id(obj):x} is calling an instrumented method {func}. "
                    "The path of this call may be incorrect."
                )

                try:
                    _id, f, path = next(iter(self._get_methods_for_func(func)))
                except Exception:
                    logger.warning(
                        "No other objects use this function so cannot guess path."
                    )
                    return None

                logger.warning(
                    f"Guessing path of new object is {path} based on other object (0x{_id:x}) using this function."
                )

                return path

            else:

                return funcs.get(func)

    def json(self, *args, **kwargs):
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json_str_of_obj(self.dict(), *args, **kwargs)

    def dict(self):
        # Same problem as in json.
        return jsonify(self, instrument=self.instrument)

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
    def _on_new_record(self, func) -> Iterable[RecordingContext]:
        ctx = self.recording_contexts.get(contextvars.Token.MISSING)

        while ctx is not contextvars.Token.MISSING:
            yield ctx
            ctx = ctx.token.old_value

    # WithInstrumentCallbacks requirement
    def _on_add_record(
        self, ctx: RecordingContext, func: Callable, sig: Signature,
        bindings: BoundArguments, ret: Any, error: Any, perf: Perf, cost: Cost
    ) -> Record:
        """
        Called by instrumented methods if they use _new_record to construct a
        record call list. 
        """

        def build_record(calls, record_metadata):
            assert len(calls) > 0, "No information recorded in call."

            main_in = self.main_input(func, sig, bindings)
            main_out = self.main_output(func, sig, bindings, ret)

            return Record(
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

        # tp = TP()

        # Finishing record needs to be done in a thread lock, done there:
        record = ctx.finish_record(build_record)

        if error is not None:
            # May block on DB.
            self._handle_error(record=record, error=error)
            raise error

        # Will block on DB, but not on feedback evaluation, depending on
        # FeedbackMode:
        record.feedback_results = self._handle_record(record=record)

        if record.feedback_results is None:
            return record

        # If in blocking mode ("WITH_APP"), wait for feedbacks to finished
        # evaluating before returning the record.
        if self.feedback_mode in [FeedbackMode.WITH_APP]:
            futures.wait(record.feedback_results)

        return record

    def _check_instrumented(self, func):
        """
        Issue a warning and some instructions if a function that has not been
        instrumented is being used in a `with_` call.
        """

        if not hasattr(func, "__name__"):
            if hasattr(func, "__call__"):
                func = func.__call__
            else:
                raise TypeError(
                    f"Unexpected type of callable `{type(func).__name__}`."
                )

        if not hasattr(func, Instrument.INSTRUMENT):
            logger.warning(
                f"Function `{func.__name__}` has not been instrumented. "
                f"This may be ok if it will call a function that has been instrumented exactly once. "
                f"Otherwise unexpected results may follow. "
                f"You can use `AddInstruments.method` of `trulens_eval.instruments` before you use the `{self.__class__.__name__}` wrapper "
                f"to make sure `{func.__name__}` does get instrumented. "
                f"`{self.__class__.__name__}` method `print_instrumented` may be used to see methods that have been instrumented. "
            )

    async def awith_(self, func, *args, **kwargs) -> Any:
        """
        Call the given async `func` with the given `*args` and `**kwargs` while
        recording, producing `func` results. The record of the computation is
        available through other means like the database or dashboard. If you
        need a record of this execution immediately, you can use `awith_record`
        or the `App` as a context mananger instead.
        """

        self._check_instrumented(func)

        res, _ = await self.awith_record(func, *args, **kwargs)

        return res

    async def awith_record(
        self,
        func,
        *args,
        record_metadata: JSON = None,
        **kwargs
    ) -> Tuple[Any, Record]:
        """
        Call the given async `func` with the given `*args` and `**kwargs`,
        producing its results as well as a record of the execution.
        """

        self._check_instrumented(func)

        with self as ctx:
            ctx.record_metadata = record_metadata
            ret = await func(*args, **kwargs)

        assert len(ctx.records) > 0, (
            f"Did not create any records. "
            f"This means that no instrumented methods were invoked in the process of calling {func}."
        )

        return ret, ctx.get()

    def with_(self, func, *args, **kwargs) -> Any:
        """
        Call the given `func` with the given `*args` and `**kwargs` while
        recording, producing `func` results. The record of the computation is
        available through other means like the database or dashboard.  If you
        need a record of this execution immediately, you can use `awith_record`
        or the `App` as a context mananger instead.
        """

        self._check_instrumented(func)

        res, _ = self.with_record(func, *args, **kwargs)
        return res

    def with_record(self,
                    func,
                    *args,
                    record_metadata: JSON = None,
                    **kwargs) -> Tuple[Any, Record]:
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

    def _with_dep_message(self, method, is_async=False, with_record=False):
        # Deprecation message for the various methods that pass through to
        # wrapped app while recording.

        # TODO: enable dep message in 0.12.0

        cname = self.__class__.__name__

        iscall = method == "__call__"

        old_method = f"""{method}{"_with_record" if with_record else ""}"""
        if iscall:
            old_method = f"""call{"_with_record" if with_record else ""}"""
        new_method = f"""{"a" if is_async else ""}with_{"record" if with_record else ""}"""

        app_callable = f"""app.{method}"""
        if iscall:
            app_callable = f"app"

        print(
            f"""
`{old_method}` will be deprecated soon; To record results of your app's execution, use one of these options to invoke your app:
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

    def _add_future_feedback(self, future: 'Future[Feedback, FeedbackResult]'):
        _, res = future.result()
        self.tru.add_feedback(res)

    def _handle_record(
        self, record: Record
    ) -> Optional[List['Future[Tuple[Feedback, FeedbackResult]]']]:
        """
        Write out record-related info to database if set and schedule feedback
        functions to be evaluated.
        """

        if self.tru is None or self.feedback_mode is None:
            return None

        self.tru: Tru
        self.db: DB

        # Need to add record to db before evaluating feedback functions.
        record_id = self.tru.add_record(record=record)

        if len(self.feedbacks) == 0:
            return []

        # Add empty (to run) feedback to db.
        if self.feedback_mode == FeedbackMode.DEFERRED:
            for f in self.feedbacks:
                self.db.insert_feedback(
                    FeedbackResult(
                        name=f.name,
                        record_id=record_id,
                        feedback_definition_id=f.feedback_definition_id
                    )
                )

            return None

        elif self.feedback_mode in [FeedbackMode.WITH_APP,
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

    def instrumented(self,) -> Iterable[Tuple[JSONPath, ComponentView]]:
        """
        Enumerate instrumented components and their categories.
        """

        for q, c in instrumented_component_views(self.dict()):
            # Add the chain indicator so the resulting paths can be specified
            # for feedback selectors.
            q = JSONPath(
                path=(GetItemOrAttribute(item_or_attribute="__app__"),) + q.path
            )
            yield q, c

    def print_instrumented(self) -> None:
        print("Components:")
        self.print_instrumented_components()
        print("\nMethods:")
        self.print_instrumented_methods()

    def format_instrumented_methods(self) -> None:
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
        """
        Print instrumented components and their categories.
        """

        object_strings = []

        for t in self.instrumented():
            path = JSONPath(t[0].path[1:])
            obj = next(iter(path.get(self)))
            object_strings.append(
                f"\t{type(obj).__name__} ({t[1].__class__.__name__}) at 0x{id(obj):x} with path {str(t[0])}"
            )

        print("\n".join(object_strings))


class TruApp(App):

    def __init__(self, *args, **kwargs):
        # Since 0.2.0
        logger.warning(
            "Class TruApp is deprecated, "
            "use trulens_eval.app.App instead."
        )
        super().__init__(*args, **kwargs)

"""
Generalized root type for various libraries like llama_index and langchain .
"""

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
import traceback
from typing import (
    Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple, Type
)

import pydantic
from pydantic import Field

from trulens_eval.db import DB
from trulens_eval.feedback import Feedback
from trulens_eval.feedback.provider.endpoint import Endpoint
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
from trulens_eval.util import all_objects
from trulens_eval.util import callable_name
from trulens_eval.util import Class
from trulens_eval.util import CLASS_INFO
from trulens_eval.util import GetItemOrAttribute
from trulens_eval.util import JSON
from trulens_eval.util import JSON_BASES
from trulens_eval.util import JSON_BASES_T
from trulens_eval.util import json_str_of_obj
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.util import safe_signature
from trulens_eval.util import SerialModel
from trulens_eval.util import TP

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# App component.
COMPONENT = Any

# Component category.
# TODO: Enum
COMPONENT_CATEGORY = str


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
    # llama_index ???

    @property
    @abstractmethod
    def model_name(self) -> str:
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


class App(AppDefinition, SerialModel, WithInstrumentCallbacks):
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

        super().__init__(**kwargs)

        self.instrument.instrument_object(
            obj=self.app, query=Select.Query().app
        )

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
                logger.warn(
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

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """

        all_args = list(bindings.arguments.values())

        # If there is only one string arg, it is a pretty good guess that it is
        # the main input.
        if len(all_args) == 1 and isinstance(all_args[0], str):
            return all_args[0]

        # Otherwise we are not sure.
        logger.warning(
            f"Unsure what the main input string is for the call to {callable_name(func)}."
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

        logger.warning(
            f"Unsure what the main output string is for the call to {callable_name(func)}."
        )

        if isinstance(ret, Dict):
            return next(iter(ret.values()))

        elif isinstance(ret, Sequence):
            if len(ret) > 0:
                return ret[0]
            else:
                return None

        else:
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

        funcs = self.instrumented_methods.get(id(obj))

        if funcs is None:
            logger.warning(
                f"A new object of type {type(obj)} at 0x{id(obj):x} is calling an instrumented method {func}. The path of this call may be incorrect."
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
                    f"A new object of type {type(obj)} at 0x{id(obj):x} is calling an instrumented method {func}. The path of this call may be incorrect."
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

    """
    # TODO: ROOTLESS

    # WithInstrumentCallbacks requirement
    def _on_new_record(self, func):
        if self.records.get(None) is not None:
            return []

        return None

    # WithInstrumentCallbacks requirement
    def _on_add_record(
        self, record: Sequence[RecordAppCall], func: Callable, sig: Signature,
        bindings: BoundArguments, start_time, end_time, ret: Any, error: Any,
        cost: Cost
    ):
        assert self.records.get(
            None
        ) is not None, "App was not expecting to keep track of a record."

        if len(record) == 0:
            logger.warning(
                "No intrumented methods called. "
                "This may be due to missing instrumentation of relevant methods. "
                f"Methods currently instrumented are: {list(self.instrumented_methods.keys())}"
            )
            raise RuntimeError("Empty record.")

        main_in = self.main_input(func, sig, bindings)
        main_out = self.main_output(func, sig, bindings, ret)

        ret_record_args = dict()
        ret_record_args['main_input'] = jsonify(main_in)

        if ret is not None:
            ret_record_args['main_output'] = jsonify(main_out)

        if error is not None:
            ret_record_args['main_error'] = jsonify(error)

        ret_record = self._post_record(
            ret_record_args, error, cost, start_time, end_time, record
        )

        records = self.records.get()
        records += [ret_record]
    """

    async def awith_record(self, func, *args, **kwargs) -> Tuple[Any, Record]:
        """
        Call the given instrumented async function `func` with the given `args`,
        `kwargs`, producing its results as well as a record.
        """

        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads. We also store self there are multiple apps
        # may have instrumented the same methods.
        record = []
        # DO NOT REMOVE
        record_and_app: Tuple[Sequence[RecordAppCall], App] = (record, self)

        ret = None
        error = None

        cost: Cost = Cost()

        start_time = None
        end_time = None

        main_in = None
        main_out = None

        start_time = datetime.now()

        try:
            sig = safe_signature(func)
            bindings = sig.bind(*args, **kwargs)

            main_in = self.main_input(func, sig, bindings)

            ret, cost = await Endpoint.atrack_all_costs_tally(
                lambda: func(*bindings.args, **bindings.kwargs)
            )

            main_out = self.main_output(func, sig, bindings, ret)

        except BaseException as e:
            error = e
            logger.error(f"App raised an exception: {e}")
            logger.error(traceback.format_exc())

        finally:
            end_time = datetime.now()

        if len(record) == 0:
            logger.warning(
                "No intrumented methods called. "
                "This may be due to missing instrumentation of relevant methods. "
                f"Methods currently instrumented are: {list(self.instrumented_methods.keys())}"
            )
            raise RuntimeError("Empty record.")

        ret_record_args = dict()

        ret_record_args['main_input'] = jsonify(main_in)

        if ret is not None:
            ret_record_args['main_output'] = jsonify(main_out)

        if error is not None:
            ret_record_args['main_error'] = jsonify(error)

        perf = Perf(start_time=start_time, end_time=end_time)
        ret_record = self._post_record(
            ret_record_args, error, cost, perf, record
        )

        return ret, ret_record

    def with_record(self, func, *args, **kwargs) -> Tuple[Any, Record]:
        """
        Call the given instrumented function `func` with the given `args`,
        `kwargs`, producing its results as well as a record.
        """

        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads. We also store self there are multiple apps
        # may have instrumented the same methods.
        record = []
        # DO NOT REMOVE
        record_and_app: Tuple[Sequence[RecordAppCall], App] = (record, self)

        ret = None
        error = None

        cost: Cost = Cost()

        start_time = None
        end_time = None

        main_in = None
        main_out = None

        start_time = datetime.now()

        try:
            sig = safe_signature(func)

            bindings = sig.bind(*args, **kwargs)

            main_in = self.main_input(func, sig, bindings)

            ret, cost = Endpoint.track_all_costs_tally(
                lambda: func(*bindings.args, **bindings.kwargs)
            )

            main_out = self.main_output(func, sig, bindings, ret)

        except BaseException as e:
            error = e
            logger.error(f"App raised an exception: {e}")
            logger.error(traceback.format_exc())
        finally:
            end_time = datetime.now()

        if len(record) == 0:
            logger.warning(
                "No intrumented methods called. "
                "This may be due to missing instrumentation of relevant methods. "
                f"Methods currently instrumented are: \n{self.format_instrumented_methods()}"
            )
            raise RuntimeError("Empty record.")

        ret_record_args = dict()

        ret_record_args['main_input'] = jsonify(main_in)

        if ret is not None:
            ret_record_args['main_output'] = jsonify(main_out)

        if error is not None:
            ret_record_args['main_error'] = jsonify(error)

        perf = Perf(start_time=start_time, end_time=end_time)
        ret_record = self._post_record(
            ret_record_args=ret_record_args,
            error=error,
            cost=cost,
            perf=perf,
            calls=record
        )

        return ret, ret_record

    def json(self, *args, **kwargs):
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json_str_of_obj(self.dict(), *args, **kwargs)

    def dict(self):
        # Same problem as in json.
        return jsonify(self, instrument=self.instrument)

    def _post_record(
        self, *, ret_record_args: Dict, error: Optional[Exception], cost: Cost,
        perf: Perf, calls: Sequence[RecordAppCall]
    ):
        """
        Final steps of record construction common among model types.
        """

        ret_record_args['main_error'] = str(error)
        ret_record_args['calls'] = calls
        ret_record_args['cost'] = cost
        ret_record_args['perf'] = perf
        ret_record_args['app_id'] = self.app_id
        ret_record_args['tags'] = self.tags

        ret_record = Record(**ret_record_args)

        if error is not None:
            if self.feedback_mode == FeedbackMode.WITH_APP:
                self._handle_error(record=ret_record, error=error)

            elif self.feedback_mode in [FeedbackMode.DEFERRED,
                                        FeedbackMode.WITH_APP_THREAD]:
                TP().runlater(
                    self._handle_error, record=ret_record, error=error
                )

            raise error

        if self.feedback_mode == FeedbackMode.WITH_APP:
            self._handle_record(record=ret_record)

        elif self.feedback_mode in [FeedbackMode.DEFERRED,
                                    FeedbackMode.WITH_APP_THREAD]:
            TP().runlater(self._handle_record, record=ret_record)

        return ret_record

    def _handle_record(self, record: Record):
        """
        Write out record-related info to database if set.
        """

        if self.tru is None or self.feedback_mode is None:
            return

        record_id = self.tru.add_record(record=record)

        if len(self.feedbacks) == 0:
            return

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

        elif self.feedback_mode in [FeedbackMode.WITH_APP,
                                    FeedbackMode.WITH_APP_THREAD]:

            results = self.tru.run_feedback_functions(
                record=record, feedback_functions=self.feedbacks, app=self
            )

            for result in results:
                self.tru.add_feedback(result)

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
        Print instrumented components and their categories.
        """
        print(self.format_instrumented_methods())

    def print_instrumented_components(self) -> None:
        """
        Print instrumented components and their categories.
        """

        print(
            "\n".join(
                f"{t[1].__class__.__name__} of {t[1].__class__.__module__} component: "
                f"{str(t[0])}" for t in self.instrumented()
            )
        )


class TruApp(App):

    def __init__(self, *args, **kwargs):
        # Since 0.2.0
        logger.warning(
            "Class TruApp is deprecated, "
            "use trulens_eval.app.App instead."
        )
        super().__init__(*args, **kwargs)

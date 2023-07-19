"""
Generalized root type for various libraries like llama_index and langchain .
"""

from abc import ABC
from abc import abstractmethod
import logging
from pprint import PrettyPrinter
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
)

import pydantic
from pydantic import Field

from trulens_eval.db import DB
from trulens_eval.feedback import Feedback
from trulens_eval.instruments import Instrument
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import Perf
from trulens_eval.schema import Record
from trulens_eval.schema import Select
from trulens_eval.tru import Tru
from trulens_eval.util import all_objects
from trulens_eval.util import Class
from trulens_eval.util import CLASS_INFO
from trulens_eval.util import GetItemOrAttribute
from trulens_eval.util import JSON
from trulens_eval.util import JSON_BASES
from trulens_eval.util import JSON_BASES_T
from trulens_eval.util import json_str_of_obj
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
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
        else:
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


class LangChainComponent(ComponentView):

    @staticmethod
    def class_is(cls: Class) -> bool:
        if cls.module.module_name.startswith("langchain."):
            return True

        #if any(base.module.module_name.startswith("langchain.")
        #       for base in cls.bases):
        #    return True

        return False

    @staticmethod
    def of_json(json: JSON) -> 'LangChainComponent':
        from trulens_eval.utils.langchain import component_of_json
        return component_of_json(json)


class LlamaIndexComponent(ComponentView):

    @staticmethod
    def class_is(cls: Class) -> bool:
        if cls.module.module_name.startswith("llama_index."):
            return True

        #if any(base.module.module_name.startswith("llama_index.")
        #       for base in cls.bases):
        #    return True

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
        if cls.module.module_name.startswith("trulens_eval."):
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


class App(AppDefinition, SerialModel):
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

        if tru is None:
            if self.feedback_mode != FeedbackMode.NONE:
                logger.debug("Creating default tru.")
                tru = Tru()
        else:
            if self.feedback_mode == FeedbackMode.NONE:
                logger.warn(
                    "`tru` is specified but `feedback_mode` is FeedbackMode.NONE. "
                    "No feedback evaluation and logging will occur."
                )

        self.tru = tru
        if self.tru is not None:
            self.db = tru.db

            if self.feedback_mode != FeedbackMode.NONE:
                logger.debug(
                    "Inserting app and feedback function definitions to db."
                )
                self.db.insert_app(app=self)
                for f in self.feedbacks:
                    self.db.insert_feedback_definition(f)

        else:
            if len(feedbacks) > 0:
                raise ValueError(
                    "Feedback logging requires `tru` to be specified."
                )

        if self.feedback_mode == FeedbackMode.DEFERRED:
            for f in feedbacks:
                # Try to load each of the feedback implementations. Deferred
                # mode will do this but we want to fail earlier at app
                # constructor here.
                f.implementation.load()

        self.instrument.instrument_object(
            obj=self.app, query=Select.Query().app
        )

    def json(self, *args, **kwargs):
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json_str_of_obj(self.dict(), *args, **kwargs)

    def dict(self):
        # Same problem as in json.
        return jsonify(self, instrument=self.instrument)

    def _post_record(
        self, ret_record_args, error, cost, start_time, end_time, record
    ):
        """
        Final steps of record construction common among model types.
        """

        ret_record_args['main_error'] = str(error)
        ret_record_args['calls'] = record
        ret_record_args['cost'] = cost
        ret_record_args['perf'] = Perf(start_time=start_time, end_time=end_time)
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

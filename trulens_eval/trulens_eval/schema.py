"""
# Serializable Classes

Only put classes which can be serialized in this file.

## Classes with non-serializable variants

Many of the classes defined here extending SerialModel are meant to be
serialized into json. Most are extended with non-serialized fields in other files.

Serializable       | Non-serializable
-------------------+------------------------
AppDefinition      | App, TruChain, TruLlama
FeedbackDefinition | Feedback

AppDefinition.app is the JSONized version of a wrapped app while App.app is the
actual wrapped app. We can thus inspect the contents of a wrapped app without
having to construct it. Additionally, JSONized objects like AppDefinition.app
feature information about the encoded object types in the dictionary under the
util.py:CLASS_INFO key.
"""

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from enum import Enum
import logging
from typing import Any, ClassVar, Dict, Optional, Sequence, TypeVar, Union

from munch import Munch as Bunch
import pydantic

from trulens_eval.util import Class
from trulens_eval.util import Function
from trulens_eval.util import FunctionOrMethod
from trulens_eval.util import GetItemOrAttribute
from trulens_eval.util import JSON
from trulens_eval.util import JSONPath
from trulens_eval.util import Method
from trulens_eval.util import obj_id_of_obj
from trulens_eval.util import SerialModel
from trulens_eval.util import WithClassInfo

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Identifier types.

RecordID = str
AppID = str
Tags = str
Metadata = dict
FeedbackDefinitionID = str
FeedbackResultID = str

# Serialization of python objects/methods. Not using pickling here so we can
# inspect the contents a little better before unserializaing.

# Record related:


class RecordAppCallMethod(SerialModel):
    path: JSONPath
    method: Method


class Cost(SerialModel):
    # Number of requests.
    n_requests: int = 0

    # Number of successful ones.
    n_successful_requests: int = 0

    # Number of class scores retrieved.
    n_classes: int = 0

    # Total tokens processed.
    n_tokens: int = 0

    # Number of prompt tokens supplied.
    n_prompt_tokens: int = 0

    # Number of completion tokens generated.
    n_completion_tokens: int = 0

    # Cost in USD.
    cost: float = 0.0

    def __add__(self, other: 'Cost') -> 'Cost':
        kwargs = {}
        for k in self.__fields__.keys():
            kwargs[k] = getattr(self, k) + getattr(other, k)
        return Cost(**kwargs)

    def __radd__(self, other: 'Cost') -> 'Cost':
        # Makes sum work on lists of Cost.

        if other == 0:
            return self

        return self.__add__(other)


class Perf(SerialModel):
    start_time: datetime
    end_time: datetime

    @property
    def latency(self):
        return self.end_time - self.start_time


class RecordAppCall(SerialModel):
    """
    Info regarding each instrumented method call is put into this container.
    """

    # Call stack but only containing paths of instrumented apps/other objects.
    stack: Sequence[RecordAppCallMethod]

    # Arguments to the instrumented method.
    args: JSON

    # Returns of the instrumented method if successful. Sometimes this is a
    # dict, sometimes a sequence, and sometimes a base value.
    rets: Optional[Any] = None

    # Error message if call raised exception.
    error: Optional[str] = None

    # Timestamps tracking entrance and exit of the instrumented method.
    perf: Optional[Perf] = None

    # Process id.
    pid: int

    # Thread id.
    tid: int

    def top(self):
        return self.stack[-1]

    def method(self):
        return self.top().method


class Record(SerialModel):
    record_id: RecordID
    app_id: AppID

    cost: Optional[Cost] = None  # pydantic.Field(default_factory=Cost)
    perf: Optional[Perf] = None  # pydantic.Field(default_factory=Perf)

    ts: datetime = pydantic.Field(default_factory=lambda: datetime.now())

    tags: Optional[str] = ""

    main_input: Optional[JSON] = None
    main_output: Optional[JSON] = None  # if no error
    main_error: Optional[JSON] = None  # if error

    # The collection of calls recorded. Note that these can be converted into a
    # json structure with the same paths as the app that generated this record
    # via `layout_calls_as_app`.
    calls: Sequence[RecordAppCall] = []

    def __init__(self, record_id: Optional[RecordID] = None, **kwargs):
        super().__init__(record_id="temporary", **kwargs)

        if record_id is None:
            record_id = obj_id_of_obj(self.dict(), prefix="record")

        self.record_id = record_id

    def layout_calls_as_app(self) -> JSON:
        """
        Layout the calls in this record into the structure that follows that of
        the app that created this record. This uses the paths stored in each
        `RecordAppCall` which are paths into the app.

        Note: We cannot create a validated schema.py:AppDefinitionclass (or
        subclass) object here as the layout of records differ in these ways:

            - Records do not include anything that is not an instrumented method
              hence have most of the structure of a app missing.
        
            - Records have RecordAppCall as their leafs where method definitions
              would be in the AppDefinitionstructure.
        """

        # TODO: problem: collissions
        ret = Bunch(**self.dict())

        for call in self.calls:
            frame_info = call.top(
            )  # info about the method call is at the top of the stack
            path = frame_info.path._append(
                GetItemOrAttribute(item_or_attribute=frame_info.method.name)
            )  # adds another attribute to path, from method name
            # TODO: append if already there
            ret = path.set(obj=ret, val=call)

        return ret


# Feedback related:


class Select:

    # Typing for type hints.
    Query: type = JSONPath

    # Instance for constructing queries for record json like `Record.app.llm`.
    Record: Query = Query().__record__

    # Instance for constructing queries for app json.
    App: Query = Query().__app__

    # A App's main input and main output.
    # TODO: App input/output generalization.
    RecordInput: Query = Record.main_input
    RecordOutput: Query = Record.main_output

    RecordCalls: Query = Record.app

    def for_record(query: Query) -> Query:
        return Select.Query(path=Select.Record.path + query.path)

    def for_app(query: Query) -> Query:
        return Select.Query(path=Select.App.path + query.path)

    def render_for_dashboard(query: Query) -> str:
        """
        Render the given query for use in dashboard to help user specify
        feedback functions.
        """

        if len(query) == 0:
            return "Select.Query()"

        ret = ""
        rest = None

        if query.path[0:2] == Select.RecordInput.path:
            ret = "Select.RecordInput"
            rest = query.path[2:]
        elif query.path[0:2] == Select.RecordOutput.path:
            ret = "Select.RecordOutput"
            rest = query.path[2:]
        elif query.path[0:2] == Select.RecordCalls.path:
            ret = "Select.RecordCalls"
            rest = query.path[2:]
        elif query.path[0] == Select.Record.path[0]:
            ret = "Select.Record"
            rest = query.path[1:]
        elif query.path[0] == Select.App.path[0]:
            ret = "Select.App"
            rest = query.path[1:]
        else:
            rest = query.path

        for step in rest:
            ret += repr(step)

        return f"{ret}"


# To deprecate in 1.0.0:
Query = Select


class FeedbackResultStatus(Enum):
    NONE = "none"
    RUNNING = "running"
    FAILED = "failed"
    DONE = "done"


class FeedbackCall(SerialModel):
    args: Dict[str, str]
    ret: float

    # New in 0.6.0: Any additional data a feedback function returns to display
    # alongside its float result.
    meta: Dict[str, Any] = pydantic.Field(default_factory=dict)


class FeedbackResult(SerialModel):
    feedback_result_id: FeedbackResultID

    record_id: RecordID

    feedback_definition_id: Optional[FeedbackDefinitionID] = None

    # "last timestamp"
    last_ts: datetime = pydantic.Field(default_factory=lambda: datetime.now())

    status: FeedbackResultStatus = FeedbackResultStatus.NONE

    cost: Cost = pydantic.Field(default_factory=Cost)

    name: str

    calls: Sequence[FeedbackCall] = []
    result: Optional[
        float] = None  # final result, potentially aggregating multiple calls

    error: Optional[str] = None  # if there was an error

    def __init__(
        self, feedback_result_id: Optional[FeedbackResultID] = None, **kwargs
    ):

        super().__init__(feedback_result_id="temporary", **kwargs)

        if feedback_result_id is None:
            feedback_result_id = obj_id_of_obj(
                self.dict(), prefix="feedback_result"
            )

        self.feedback_result_id = feedback_result_id


class FeedbackDefinition(SerialModel):
    # Serialized parts of a feedback function. The non-serialized parts are in
    # the feedback.py:Feedback class.

    # Implementation serialization info.
    implementation: Optional[Union[Function, Method]] = None

    # Aggregator method for serialization.
    aggregator: Optional[Union[Function, Method]] = None

    # Id, if not given, unique determined from _json below.
    feedback_definition_id: FeedbackDefinitionID

    # Selectors, pointers into Records of where to get
    # arguments for `imp`.
    selectors: Dict[str, JSONPath]

    def __init__(
        self,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        implementation: Optional[Union[Function, Method]] = None,
        aggregator: Optional[Union[Function, Method]] = None,
        selectors: Dict[str, JSONPath] = None
    ):
        """
        - selectors: Optional[Dict[str, JSONPath]] -- mapping of implementation
          argument names to where to get them from a record.

        - feedback_definition_id: Optional[str] - unique identifier.

        - implementation: Optional[Union[Function, Method]] -- the serialized
          implementation function.

        - aggregator: Optional[Union[Function, Method]] -- serialized
          aggregation function.
        """

        selectors = selectors or dict()

        super().__init__(
            feedback_definition_id="temporary",
            selectors=selectors,
            implementation=implementation,
            aggregator=aggregator,
        )

        if feedback_definition_id is None:
            if implementation is not None:
                feedback_definition_id = obj_id_of_obj(
                    self.dict(), prefix="feedback_definition"
                )
            else:
                feedback_definition_id = "anonymous_feedback_definition"

        self.feedback_definition_id = feedback_definition_id


# App related:


class FeedbackMode(str, Enum):
    # No evaluation will happen even if feedback functions are specified.
    NONE = "none"

    # Try to run feedback functions immediately and before app returns a
    # record.
    WITH_APP = "with_app"

    # Try to run feedback functions in the same process as the app but after
    # it produces a record.
    WITH_APP_THREAD = "with_app_thread"

    # Evaluate later via the process started by
    # `tru.start_deferred_feedback_evaluator`.
    DEFERRED = "deferred"


class AppDefinition(SerialModel, WithClassInfo, ABC):
    # Serialized fields here whereas app.py:App contains
    # non-serialized fields.

    class Config:
        arbitrary_types_allowed = True

    app_id: AppID
    tags: Tags
    metadata: Metadata

    # Feedback functions to evaluate on each record. Unlike the above, these are
    # meant to be serialized.
    feedback_definitions: Sequence[FeedbackDefinition] = []

    # NOTE: Custom feedback functions cannot be run deferred and will be run as
    # if "withappthread" was set.
    feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD

    # Class of the main instrumented object.
    root_class: Class  # TODO: make classvar

    # App's main method. To be filled in by subclass. Want to make this abstract
    # but this causes problems when trying to load an AppDefinition from json.
    root_callable: ClassVar[FunctionOrMethod]

    # Wrapped app in jsonized form.
    app: JSON

    def __init__(
        self,
        app_id: Optional[AppID] = None,
        tags: Optional[Tags] = None,
        metadata: Optional[Metadata] = None,
        feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD,
        **kwargs
    ):

        # for us:
        kwargs['app_id'] = "temporary"  # will be adjusted below
        kwargs['feedback_mode'] = feedback_mode
        kwargs['tags'] = ""
        kwargs['metadata'] = {}

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(**kwargs)

        if app_id is None:
            app_id = obj_id_of_obj(obj=self.dict(), prefix="app")

        self.app_id = app_id

        if tags is None:
            tags = "-"  # Set tags to a "-" if None is provided
        self.tags = tags

        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @classmethod
    def select_inputs(cls) -> JSONPath:
        """
        Get the path to the main app's call inputs.
        """

        return getattr(
            Select.RecordCalls,
            cls.root_callable.default_factory().name
        ).args

    @classmethod
    def select_outputs(cls) -> JSONPath:
        """
        Get the path to the main app's call outputs.
        """

        return getattr(
            Select.RecordCalls,
            cls.root_callable.default_factory().name
        ).rets


class App(AppDefinition):

    def __init__(self, *args, **kwargs):
        # Since 0.2.0
        logger.warning(
            "Class trulens_eval.schema.App is deprecated, "
            "use trulens_eval.schema.AppDefinition instead."
        )
        super().__init__(*args, **kwargs)

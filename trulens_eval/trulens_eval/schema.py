"""
# Serializable Classes

Only put classes which can be serialized in this file.

## Classes with non-serializable variants

Many of the classes defined here extending SerialModel are meant to be
serialized into json. Most are extended with non-serialized fields in other files.

Serializable       | Non-serializable
-------------------+---------------------------
AppDefinition               | TruApp, TruChain, TruLlama
FeedbackDefinition | Feedback

App.app is the JSONized version of a wrapped app while TruApp.app is the actual
wrapped app. We can thus inspect the contents of a wrapped app without having to
construct it. Additionally, JSONized objects like App.app feature information
about the encoded object types in the dictionary under the util.py:CLASS_INFO key.

"""

from datetime import datetime
from enum import Enum

from typing import (Any, Dict, Optional, Sequence, TypeVar, Union)
import logging
from munch import Munch as Bunch
import pydantic

from trulens_eval.util import Class
from trulens_eval.util import Function
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
    n_requests: Optional[int] = None

    # Number of successful ones.
    n_succesful_requests: Optional[int] = None

    # Number of class scores retrieved.
    n_classes: Optional[int] = None

    # Total tokens processed.
    n_tokens: Optional[int] = None

    # Number of prompt tokens supplied.
    n_prompt_tokens: Optional[int] = None

    # Number of completion tokens generated.
    n_completion_tokens: Optional[int] = None

    # Cost in USD.
    cost: Optional[float] = None


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
    perf: Perf = pydantic.Field(default_factory=Perf)

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

    tags: str = ""

    main_input: Optional[str] = None
    main_output: Optional[str] = None  # if no error
    main_error: Optional[str] = None  # if error

    # The collection of calls recorded. Note that these can be converted into a
    # json structure with the same paths as the app that generated this record
    # via `layout_calls_as_app`.
    calls: Sequence[RecordAppCall] = []

    def __init__(self, record_id: Optional[RecordID] = None, **kwargs):
        super().__init__(record_id="temporay", **kwargs)

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


class Query:

    # Typing for type hints.
    Query = JSONPath

    # Instance for constructing queries for record json like `Record.app.llm`.
    Record = Query().__record__

    # Instance for constructing queries for app json.
    App = Query().__app__

    # A App's main input and main output.
    # TODO: App input/output generalization.
    RecordInput = Record.main_input
    RecordOutput = Record.main_output


class FeedbackResultStatus(Enum):
    NONE = "none"
    RUNNING = "running"
    FAILED = "failed"
    DONE = "done"


class FeedbackCall(SerialModel):
    args: Dict[str, str]
    ret: float


class FeedbackResult(SerialModel):
    feedback_result_id: FeedbackResultID

    record_id: RecordID

    feedback_definition_id: Optional[FeedbackDefinitionID] = None

    # "last timestamp"
    last_ts: datetime = pydantic.Field(default_factory=lambda: datetime.now())

    status: FeedbackResultStatus = FeedbackResultStatus.NONE

    cost: Cost = pydantic.Field(default_factory=Cost)

    tags: str = ""

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
    selectors: Optional[Dict[str, JSONPath]] = None

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


class AppDefinition(SerialModel, WithClassInfo):
    # Serialized fields here whereas app.py:App contains
    # non-serialized fields.

    class Config:
        arbitrary_types_allowed = True

    app_id: AppID

    # Feedback functions to evaluate on each record. Unlike the above, these are
    # meant to be serialized.
    feedback_definitions: Sequence[FeedbackDefinition] = []

    # NOTE: Custom feedback functions cannot be run deferred and will be run as
    # if "withappthread" was set.
    feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD

    # Class of the main instrumented object.
    root_class: Class

    # Wrapped app in jsonized form.
    app: JSON

    def __init__(
        self,
        app_id: Optional[AppID] = None,
        feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD,
        **kwargs
    ):

        # for us:
        kwargs['app_id'] = "temporary"  # will be adjusted below
        kwargs['feedback_mode'] = feedback_mode

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(**kwargs)

        if app_id is None:
            app_id = obj_id_of_obj(obj=self.dict(), prefix="app")

        self.app_id = app_id


class App(AppDefinition):

    def __init__(self, *args, **kwargs):
        # Since 0.2.0
        logger.warning(
            "Class trulens_eval.schema.App is deprecated, "
            "use trulens_eval.schema.AppDefinition instead."
        )
        super().__init__(*args, **kwargs)

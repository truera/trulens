"""
# Serializable Classes

Only put classes which can be serialized in this file.

## Classes with non-serializable variants

Many of the classes defined here extending SerialModel are meant to be
serialized into json. Most are extended with non-serialized fields in other files.

| Serializable       | Non-serializable        |
| ------------------ | ----------------------- |
| AppDefinition      | App, TruChain, TruLlama |
| FeedbackDefinition | Feedback                |

AppDefinition.app is the JSON-ized version of a wrapped app while App.app is the
actual wrapped app. We can thus inspect the contents of a wrapped app without
having to construct it. Additionally, JSONized objects like AppDefinition.app
feature information about the encoded object types in the dictionary under the
util.py:CLASS_INFO key.
"""

from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime
from enum import Enum
import logging
from pprint import PrettyPrinter
from typing import (
    Any, Callable, ClassVar, Dict, List, Optional, Sequence, Type,
    TYPE_CHECKING, TypeVar, Union
)

import dill
import humanize
from munch import Munch as Bunch
import pydantic

from trulens_eval.utils.json import jsonify
from trulens_eval.utils.json import obj_id_of_obj
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import Function
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.pyschema import Method
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import GetItemOrAttribute
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import Lens
from trulens_eval.utils.serial import SerialBytes
from trulens_eval.utils.serial import SerialModel

T = TypeVar("T")

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

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

MAX_DILL_SIZE = 1024 * 1024  # 1MB


class RecordAppCallMethod(SerialModel):
    """
    Method information for the stacks inside `RecordAppCall`.
    """

    path: Lens
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

    # In streaming mode, number of chunks produced.
    n_stream_chunks: int = 0

    # Number of prompt tokens supplied.
    n_prompt_tokens: int = 0

    # Number of completion tokens generated.
    n_completion_tokens: int = 0

    # Cost in USD.
    cost: float = 0.0

    def __add__(self, other: 'Cost') -> 'Cost':
        kwargs = {}
        for k in self.model_fields.keys():
            kwargs[k] = getattr(self, k) + getattr(other, k)
        return Cost(**kwargs)

    def __radd__(self, other: 'Cost') -> 'Cost':
        # Makes sum work on lists of Cost.

        if other == 0:
            return self

        return self.__add__(other)


class Perf(SerialModel):
    """
    Performance information. Presently only the start and end times, and thus
    latency.
    """

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
    stack: List[RecordAppCallMethod]

    # Arguments to the instrumented method.
    args: JSON

    # Returns of the instrumented method if successful. Sometimes this is a
    # dict, sometimes a sequence, and sometimes a base value.
    rets: Optional[JSON] = None

    # Error message if call raised exception.
    error: Optional[str] = None

    # Timestamps tracking entrance and exit of the instrumented method.
    perf: Optional[Perf] = None

    # Process id.
    pid: int

    # Thread id.
    tid: int

    def top(self) -> RecordAppCallMethod:
        return self.stack[-1]

    def method(self) -> Method:
        return self.top().method


class Record(SerialModel):
    """
    Each instrumented method call produces one of these "record" instances.
    """

    model_config: ClassVar[dict] = dict(
        # for `Future[FeedbackResult]` = `TFeedbackResultFuture`
        arbitrary_types_allowed=True
    )

    record_id: RecordID  # str
    app_id: AppID  # str

    cost: Optional[Cost] = None
    perf: Optional[Perf] = None

    ts: datetime = pydantic.Field(default_factory=datetime.now)

    tags: Optional[str] = ""
    meta: Optional[JSON] = None

    main_input: Optional[JSON] = None
    main_output: Optional[JSON] = None  # if no error
    main_error: Optional[JSON] = None  # if error

    # The collection of calls recorded. Note that these can be converted into a
    # json structure with the same paths as the app that generated this record
    # via `layout_calls_as_app`.
    calls: List[RecordAppCall] = []

    # Feedback results only filled for records that were just produced. Will not
    # be filled in when read from database. Also, will not fill in when using
    # `FeedbackMode.DEFERRED`.

    feedback_results: Optional[List[TFeedbackResultFuture]] = \
        pydantic.Field(None, exclude=True)

    def __init__(self, record_id: Optional[RecordID] = None, **kwargs):
        # Fixed record_id for obj_id_of_id below.
        super().__init__(record_id="temporary", **kwargs)

        if record_id is None:
            record_id = obj_id_of_obj(jsonify(self), prefix="record")

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

        ret = Bunch(**self.model_dump())

        for call in self.calls:
            # Info about the method call is at the top of the stack
            frame_info = call.top()

            # Adds another attribute to path, from method name:
            path = frame_info.path._append(
                GetItemOrAttribute(item_or_attribute=frame_info.method.name)
            )

            ret = path.set_or_append(obj=ret, val=call)

        return ret


# Feedback related:


class Select:
    """
    Utilities for creating selectors using Lens and aliases/shortcuts.
    """

    # Typing for type hints.
    # TODEP
    Query = Lens

    # Instance for constructing queries for record json like `Record.app.llm`.
    Record: Query = Query().__record__

    # Instance for constructing queries for app json.
    App: Query = Query().__app__

    # A App's main input and main output.
    # TODO: App input/output generalization.
    RecordInput: Query = Record.main_input  # type: ignore
    RecordOutput: Query = Record.main_output  # type: ignore

    # The calls made by the wrapped app. Layed out by path into components.
    RecordCalls: Query = Record.app  # type: ignore

    # The first called method (last to return).
    RecordCall: Query = Record.calls[-1]

    # The whole set of inputs/arguments to the first called / last method call.
    RecordArgs: Query = RecordCall.args
    # The whole output of the first called / last returned method call.
    RecordRets: Query = RecordCall.rets

    @staticmethod
    def context(app: Optional[Any] = None) -> Lens:
        from trulens_eval.app import App
        return App.select_context(app)

    @staticmethod
    def for_record(query: Select.Query) -> Query:
        return Select.Query(path=Select.Record.path + query.path)

    @staticmethod
    def for_app(query: Select.Query) -> Query:
        return Select.Query(path=Select.App.path + query.path)

    @staticmethod
    def render_for_dashboard(query: Select.Query) -> str:
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

        elif query.path[0:4] == Select.RecordArgs.path:
            ret = "Select.RecordArgs"
            rest = query.path[4:]
        elif query.path[0:4] == Select.RecordRets.path:
            ret = "Select.RecordRets"
            rest = query.path[4:]

        elif query.path[0:2] == Select.RecordCalls.path:
            ret = "Select.RecordCalls"
            rest = query.path[2:]

        elif query.path[0:3] == Select.RecordCall.path:
            ret = "Select.RecordCall"
            rest = query.path[3:]

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


class FeedbackResultStatus(Enum):
    """
    For deferred feedback evaluation, these values indicate status of evaluation.
    """

    # Initial value is none.
    NONE = "none"

    # Once queued/started, status is updated to "running".
    RUNNING = "running"

    # If run failed.
    FAILED = "failed"

    # If run completed successfully.
    DONE = "done"


class FeedbackCall(SerialModel):
    """
    Invocations of feedback function results in one of these instances. Note
    that a single `Feedback` instance might require more than one call.
    """

    # Arguments to the feedback function.
    args: Dict[str, Optional[JSON]]

    # Return value.
    ret: float

    # New in 0.6.0: Any additional data a feedback function returns to display
    # alongside its float result.
    meta: Dict[str, Any] = pydantic.Field(default_factory=dict)


class FeedbackResult(SerialModel):
    """
    Feedback results for a single `Feedback` instance. This might involve
    multiple feedback function calls.
    """

    feedback_result_id: FeedbackResultID

    # Record over which the feedback was evaluated.
    record_id: RecordID

    # The `Feedback` / `FeedbackDefinition` which was evaluated to get this
    # result.
    feedback_definition_id: Optional[FeedbackDefinitionID] = None

    # Last timestamp involved in the evaluation.
    last_ts: datetime = pydantic.Field(default_factory=datetime.now)

    # For deferred feedback evaluation, the status of the evaluation.
    status: FeedbackResultStatus = FeedbackResultStatus.NONE

    cost: Cost = pydantic.Field(default_factory=Cost)

    # Given name of the feedback.
    name: str

    # Individual feedback function invocations.
    calls: Sequence[FeedbackCall] = []

    # Final result, potentially aggregating multiple calls.
    result: Optional[float] = None

    # Error information if there was an error.
    error: Optional[str] = None

    # TODO: doc
    multi_result: Optional[str] = None

    def __init__(
        self, feedback_result_id: Optional[FeedbackResultID] = None, **kwargs
    ):
        super().__init__(feedback_result_id="temporary", **kwargs)

        if feedback_result_id is None:
            feedback_result_id = obj_id_of_obj(
                self.model_dump(), prefix="feedback_result"
            )

        self.feedback_result_id = feedback_result_id


if TYPE_CHECKING:
    TFeedbackResultFuture = Future[FeedbackResult]
else:
    TFeedbackResultFuture = Future


class FeedbackDefinition(WithClassInfo, SerialModel):
    # Serialized parts of a feedback function. The non-serialized parts are in
    # the feedback.py:Feedback class.

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    # Implementation serialization info.
    implementation: Optional[Union[Function, Method]] = None

    # Aggregator method for serialization.
    aggregator: Optional[Union[Function, Method]] = None

    # Id, if not given, unique determined from _json below.
    feedback_definition_id: FeedbackDefinitionID

    # Selectors, pointers into Records of where to get
    # arguments for `imp`.
    selectors: Dict[str, Lens]

    supplied_name: Optional[str] = None

    def __init__(
        self,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        implementation: Optional[Union[Function, Method]] = None,
        aggregator: Optional[Union[Function, Method]] = None,
        selectors: Optional[Dict[str, Lens]] = None,
        supplied_name: Optional[str] = None,
        **kwargs
    ):
        """
        - selectors: Optional[Dict[str, Lens]] -- mapping of implementation
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
            supplied_name=supplied_name,
            **kwargs
        )

        if feedback_definition_id is None:
            if implementation is not None:
                feedback_definition_id = obj_id_of_obj(
                    self.model_dump(), prefix="feedback_definition"
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


class AppDefinition(WithClassInfo, SerialModel):
    # Serialized fields here whereas app.py:App contains
    # non-serialized fields.

    app_id: AppID  # str
    tags: Tags  # str
    metadata: Metadata  # dict  # TODO: rename to meta for consistency with other metas

    # Feedback functions to evaluate on each record. Unlike the above, these are
    # meant to be serialized.
    feedback_definitions: Sequence[FeedbackDefinition] = []

    # NOTE: Custom feedback functions cannot be run deferred and will be run as
    # if "withappthread" was set.
    feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD

    # Class of the main instrumented object. Ideally this would be a ClassVar
    # but since we want to check this without instantiating the subclass of
    # AppDefinition that would define it, we cannot use ClassVar.
    root_class: Class

    # App's main method. To be filled in by subclass. Want to make this abstract
    # but this causes problems when trying to load an AppDefinition from json.
    root_callable: ClassVar[FunctionOrMethod]

    # Wrapped app in jsonized form.
    app: JSON

    # EXPERIMENTAL
    # NOTE: temporary unsafe serialization of function that loads the app:
    # Dump of the initial app before any invocations. Can be used to create a new session.
    initial_app_loader_dump: Optional[SerialBytes] = None

    # Info to store about the app and to display in dashboard. This is useful if
    # app itself cannot be serialized. `app_extra_json`, then, can stand in place for
    # whatever the user might want to see about the app.
    app_extra_json: JSON

    def __init__(
        self,
        app_id: Optional[AppID] = None,
        tags: Optional[Tags] = None,
        metadata: Optional[Metadata] = None,
        feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD,
        app_extra_json: JSON = None,
        **kwargs
    ):

        # for us:
        kwargs['app_id'] = "temporary"  # will be adjusted below
        kwargs['feedback_mode'] = feedback_mode
        kwargs['tags'] = ""
        kwargs['metadata'] = {}
        kwargs['app_extra_json'] = app_extra_json or dict()
        
        super().__init__(**kwargs)

        if app_id is None:
            app_id = obj_id_of_obj(obj=self.model_dump(), prefix="app")

        self.app_id = app_id

        if tags is None:
            tags = "-"  # Set tags to a "-" if None is provided
        self.tags = tags

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        # EXPERIMENTAL
        if 'initial_app_loader' in kwargs:
            try:
                dump = dill.dumps(kwargs['initial_app_loader'], recurse=True)

                if len(dump) > MAX_DILL_SIZE:
                    logger.warning(
                        f"`initial_app_loader` dump is too big ({humanize.naturalsize(len(dump))} > {humanize.naturaldate(MAX_DILL_SIZE)} bytes). "
                        "If you are loading large objects, include the loading logic inside `initial_app_loader`."
                    )
                else:
                    self.initial_app_loader_dump = SerialBytes(data=dump)

                    # This is an older serialization approach that saved things
                    # in local files instead of the DB. Leaving here for now as
                    # serialization of large apps might make this necessary
                    # again.
                    """
                    path_json = Path.cwd() / f"{app_id}.json"
                    path_dill = Path.cwd() / f"{app_id}.dill"

                    with path_json.open("w") as fh:
                        fh.write(json_str_of_obj(self))

                    with path_dill.open("wb") as fh:
                        fh.write(dump)

                    print(f"Wrote loadable app to {path_json} and {path_dill}.")
                    """

            except Exception as e:
                logger.warning(
                    f"Could not serialize app loader. "
                    f"Some trulens features may not be available: {e}"
                )


    @staticmethod
    def continue_session(
        app_definition_json: JSON, app: Any
    ) -> 'AppDefinition':
        # initial_app_loader: Optional[Callable] = None) -> 'AppDefinition':
        """
        Create a copy of the json serialized app with the enclosed app being
        initialized to its initial state before any records are produced (i.e.
        blank memory).
        """

        app_definition_json['app'] = app

        cls = WithClassInfo.get_class(app_definition_json)

        return cls(**app_definition_json)

    @staticmethod
    def new_session(
        app_definition_json: JSON,
        initial_app_loader: Optional[Callable] = None
    ) -> 'AppDefinition':
        """
        Create a copy of the json serialized app with the enclosed app being
        initialized to its initial state before any records are produced (i.e.
        blank memory).
        """

        serial_bytes_json: Optional[JSON] = app_definition_json[
            'initial_app_loader_dump']

        if initial_app_loader is None:
            assert serial_bytes_json is not None, "Cannot create new session without `initial_app_loader`."

            serial_bytes = SerialBytes.model_validate(serial_bytes_json)

            app = dill.loads(serial_bytes.data)()

        else:
            app = initial_app_loader()
            data = dill.dumps(initial_app_loader, recurse=True)
            serial_bytes = SerialBytes(data=data)
            serial_bytes_json = serial_bytes.model_dump()

        app_definition_json['app'] = app
        app_definition_json['initial_app_loader_dump'] = serial_bytes_json

        cls: Type[App] = WithClassInfo.get_class(app_definition_json)

        return cls.model_validate_json(app_definition_json)

    def jsonify_extra(self, content):
        # Called by jsonify for us to add any data we might want to add to the
        # serialization of `app`.
        if self.app_extra_json is not None:
            content['app'].update(self.app_extra_json)

        return content

    @staticmethod
    def get_loadable_apps():
        # EXPERIMENTAL
        """
        Gets a list of all of the loadable apps. This is those that have
        `initial_app_loader_dump` set.
        """

        rets = []

        from trulens_eval import Tru

        tru = Tru()

        apps = tru.get_apps()
        for app in apps:
            dump = app.get('initial_app_loader_dump')
            if dump is not None:
                rets.append(app)

        return rets

    def dict(self):
        # Unsure if the check below is needed. Sometimes we have an `app.App`` but
        # it is considered an `AppDefinition` and is thus using this definition
        # of `dict` instead of the one in `app.App`.

        from trulens_eval.trulens_eval import app
        if isinstance(self, app.App):
            return jsonify(self, instrument=self.instrument)
        else:
            return jsonify(self)

    @classmethod
    def select_inputs(cls) -> Lens:
        """
        Get the path to the main app's call inputs.
        """

        return getattr(
            Select.RecordCalls,
            cls.root_callable.default_factory().name
        ).args

    @classmethod
    def select_outputs(cls) -> Lens:
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

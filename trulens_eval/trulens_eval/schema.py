"""
# Serializable Classes

Note: Only put classes which can be serialized in this module.

## Classes with non-serializable variants

Many of the classes defined here extending serial.SerialModel are meant to be
serialized into json. Most are extended with non-serialized fields in other files.

| Serializable       | Non-serializable        |
| ------------------ | ----------------------- |
| [AppDefinition][trulens_eval.schema.AppDefinition] | [App][trulens_eval.app.App], Tru{Chain, Llama, ...} |
| [FeedbackDefinition][trulens_eval.schema.FeedbackDefinition] | [Feedback][trulens_eval.feedback.feedback.Feedback] |

`AppDefinition.app` is the JSON-ized version of a wrapped app while `App.app` is the
actual wrapped app. We can thus inspect the contents of a wrapped app without
having to construct it. Additionally, JSONized objects like `AppDefinition.app`
feature information about the encoded object types in the dictionary under the
`util.py:CLASS_INFO` key.

"""

from __future__ import annotations

import datetime
from enum import Enum
import logging
from pprint import PrettyPrinter
from typing import (Any, Callable, ClassVar, Dict, Hashable, List, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import dill
import humanize
from munch import Munch as Bunch
import pydantic
import typing_extensions

from trulens_eval.utils import pyschema
from trulens_eval.utils import serial
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.json import obj_id_of_obj
from trulens_eval.utils.python import Future

T = TypeVar("T")

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

# Identifier type aliases.

RecordID: typing_extensions.TypeAlias = str
"""Unique identifier for a record."""

AppID: typing_extensions.TypeAlias = str
"""Unique identifier for an app."""

Tags: typing_extensions.TypeAlias = str
"""Tags for an app or record."""

Metadata: typing_extensions.TypeAlias = Dict
"""Metadata for an app or record."""

FeedbackDefinitionID: typing_extensions.TypeAlias = str
"""Unique identifier for a feedback definition."""

FeedbackResultID: typing_extensions.TypeAlias = str
"""Unique identifier for a feedback result."""

# Record related:

MAX_DILL_SIZE: int = 1024 * 1024  # 1MB
"""Max size in bytes of pickled objects."""


class RecordAppCallMethod(serial.SerialModel):
    """Method information for the stacks inside `RecordAppCall`."""

    path: serial.Lens
    """Path to the method in the app's structure."""

    method: pyschema.Method
    """The method that was called."""


class Cost(serial.SerialModel, pydantic.BaseModel):
    """Costs associated with some call or set of calls."""

    n_requests: int = 0
    """Number of requests."""

    n_successful_requests: int = 0
    """Number of successful requests."""

    n_classes: int = 0
    """Number of class scores retrieved."""

    n_tokens: int = 0
    """Total tokens processed."""

    n_stream_chunks: int = 0
    """In streaming mode, number of chunks produced."""

    n_prompt_tokens: int = 0
    """Number of prompt tokens supplied."""

    n_completion_tokens: int = 0
    """Number of completion tokens generated."""

    cost: float = 0.0
    """Cost in USD."""

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


class Perf(serial.SerialModel, pydantic.BaseModel):
    """Performance information.
    
    Presently only the start and end times, and thus latency.
    """

    start_time: datetime.datetime
    """Datetime before the recorded call."""
    
    end_time: datetime.datetime
    """Datetime after the recorded call."""

    @property
    def latency(self):
        """Latency in seconds."""
        return self.end_time - self.start_time


class RecordAppCall(serial.SerialModel):
    """Info regarding each instrumented method call."""

    stack: List[RecordAppCallMethod]
    """Call stack but only containing paths of instrumented apps/other objects."""

    args: serial.JSON
    """Arguments to the instrumented method."""

    rets: Optional[serial.JSON] = None
    """Returns of the instrumented method if successful.
    
    Sometimes this is a dict, sometimes a sequence, and sometimes a base value.
    """
    
    error: Optional[str] = None
    """Error message if call raised exception."""

    perf: Optional[Perf] = None
    """Timestamps tracking entrance and exit of the instrumented method."""

    pid: int
    """Process id."""

    tid: int
    """Thread id."""

    def top(self) -> RecordAppCallMethod:
        """The top of the stack."""

        return self.stack[-1]

    def method(self) -> pyschema.Method:
        """The method at the top of the stack."""

        return self.top().method


class Record(serial.SerialModel, Hashable):
    """Each instrumented method call produces one of these "record" instances."""

    model_config: ClassVar[dict] = dict(
        # for `Future[FeedbackResult]`
        arbitrary_types_allowed=True
    )

    record_id: RecordID
    """Unique identifier for this record."""

    app_id: AppID
    """The app that produced this record."""

    cost: Optional[Cost] = None
    """Costs associated with the record."""

    perf: Optional[Perf] = None
    """Performance information."""

    ts: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    """Timestamp of last update.
    
    This is usually set whenever a record is changed in any way."""

    tags: Optional[str] = ""
    """Tags for the record."""

    meta: Optional[serial.JSON] = None
    """Metadata for the record."""

    main_input: Optional[serial.JSON] = None
    """The app's main input."""

    main_output: Optional[serial.JSON] = None  # if no error
    """The app's main output if there was no error."""

    main_error: Optional[serial.JSON] = None  # if error
    """The app's main error if there was an error."""

    calls: List[RecordAppCall] = []
    """The collection of calls recorded.

    Note that these can be converted into a json structure with the same paths
    as the app that generated this record via `layout_calls_as_app`.
    """

    feedback_and_future_results: Optional[List[Tuple[
        FeedbackDefinition, Future[FeedbackResult]
    ]]] = pydantic.Field(None, exclude=True)
    """Map of feedbacks to the futures for of their results.
     
    These are only filled for records that were just produced. This will not
    be filled in when read from database. Also, will not fill in when using
    `FeedbackMode.DEFERRED`.
    """

    feedback_results: Optional[List[Future[FeedbackResult]]] = \
        pydantic.Field(None, exclude=True)
    """Only the futures part of the above for backwards compatibility."""

    def __init__(self, record_id: Optional[RecordID] = None, **kwargs):
        super().__init__(record_id="temporary", **kwargs)

        if record_id is None:
            record_id = obj_id_of_obj(jsonify(self), prefix="record")

        self.record_id = record_id

    def __hash__(self):
        return hash(self.record_id)

    def wait_for_feedback_results(
        self
    ) -> Dict[FeedbackDefinition, FeedbackResult]:
        """Wait for feedback results to finish.

        Returns:
            A mapping of feedback functions to their results.
        """

        if self.feedback_and_future_results is None:
            return {}

        ret = {}

        for feedback, future_result in self.feedback_and_future_results:
            feedback_result = future_result.result()
            ret[feedback] = feedback_result

        return ret

    def layout_calls_as_app(self) -> serial.JSON:
        """Layout the calls in this record into the structure that follows that of
        the app that created this record.
        
        This uses the paths stored in each `RecordAppCall` which are paths into
        the app.

        Note: We cannot create a validated `schema.py:AppDefinition` class (or
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
                serial.GetItemOrAttribute(item_or_attribute=frame_info.method.name)
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
    Query = serial.Lens

    # The tru wrapper (TruLlama, TruChain, etc.)
    Tru: Query = Query()

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
    def path_and_method(select: Select.Query) -> Tuple[Select.Query, str]:
        """
        If `select` names in method as the last attribute, extract the method name
        and the selector without the final method name.
        """

        if len(select.path) == 0:
            raise ValueError(
                "Given selector is empty so does not name a method."
            )

        firsts = select.path[:-1]
        last = select.path[-1]

        if not isinstance(last, serial.StepItemOrAttribute):
            raise ValueError(
                "Last part of selector is not an attribute so does not name a method."
            )

        method_name = last.get_item_or_attribute()
        path = Select.Query(path=firsts)

        return path, method_name

    @staticmethod
    def dequalify(select: Select.Query) -> Select.Query:
        """
        If the given selector qualifies record or app, remove that
        qualification.
        """

        if len(select.path) == 0:
            return select

        if select.path[0] == Select.Record.path[0] or \
            select.path[0] == Select.App.path[0]:
            return Select.Query(path=select.path[1:])

        return select

    @staticmethod
    def context(app: Optional[Any] = None) -> serial.Lens:
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


class FeedbackCall(serial.SerialModel):
    """
    Invocations of feedback function results in one of these instances. Note
    that a single `Feedback` instance might require more than one call.
    """

    # Arguments to the feedback function.
    args: Dict[str, Optional[serial.JSON]]

    # Return value.
    ret: float

    # New in 0.6.0: Any additional data a feedback function returns to display
    # alongside its float result.
    meta: Dict[str, Any] = pydantic.Field(default_factory=dict)


class FeedbackResult(serial.SerialModel):
    """Feedback results for a single [Feedback][trulens_eval.feedback.feedback.Feedback] instance.
    
    This might involve multiple feedback function calls. Typically you should
    not be constructing these objects yourself except for the cases where you'd
    like to log human feedback.

    Attributes:
        feedback_result_id (str): Unique identifier for this result.

        record_id (str): Record over which the feedback was evaluated.

        feedback_definition_id (str): The id of the
            [FeedbackDefinition][trulens_eval.schema.FeedbackDefinition] which
            was evaluated to get this result.

        last_ts (datetime.datetime): Last timestamp involved in the evaluation.

        status (FeedbackResultStatus): For deferred feedback evaluation, the
            status of the evaluation.

        cost (Cost): Cost of the evaluation.

        name (str): Given name of the feedback.

        calls (List[FeedbackCall]): Individual feedback function invocations.

        result (float): Final result, potentially aggregating multiple calls.

        error (str): Error information if there was an error.

        multi_result (str): TODO: doc
    """

    feedback_result_id: FeedbackResultID

    # Record over which the feedback was evaluated.
    record_id: RecordID

    # The `Feedback` / `FeedbackDefinition` which was evaluated to get this
    # result.
    feedback_definition_id: Optional[FeedbackDefinitionID] = None

    # Last timestamp involved in the evaluation.
    last_ts: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)

    status: FeedbackResultStatus = FeedbackResultStatus.NONE
    """For deferred feedback evaluation, the status of the evaluation."""

    cost: Cost = pydantic.Field(default_factory=Cost)

    # Given name of the feedback.
    name: str

    # Individual feedback function invocations.
    calls: List[FeedbackCall] = []

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


class FeedbackDefinition(pyschema.WithClassInfo, serial.SerialModel, Hashable):
    """Serialized parts of a feedback function. 
    
    The non-serialized parts are in the
    [Feedback][trulens_eval.feedback.feedback.Feedback] class.
    """

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    implementation: Optional[Union[pyschema.Function, pyschema.Method]] = None
    """Implementation serialization."""

    aggregator: Optional[Union[pyschema.Function, pyschema.Method]] = None
    """Aggregator method serialization."""

    feedback_definition_id: FeedbackDefinitionID
    """Id, if not given, uniquely determined from content."""

    selectors: Dict[str, serial.Lens]
    """Selectors; pointers into Records of where to get arguments for `imp`."""

    supplied_name: Optional[str] = None
    """An optional name. Only will affect displayed tables."""

    higher_is_better: Optional[bool] = None
    """Feedback result magnitude interpretation."""

    def __init__(
        self,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        implementation: Optional[Union[pyschema.Function, pyschema.Method]] = None,
        aggregator: Optional[Union[pyschema.Function, pyschema.Method]] = None,
        selectors: Optional[Dict[str, serial.Lens]] = None,
        name: Optional[str] = None,
        higher_is_better: Optional[bool] = None,
        **kwargs
    ):
        selectors = selectors or dict()

        if name is not None:
            kwargs['supplied_name'] = name

        super().__init__(
            feedback_definition_id="temporary",
            selectors=selectors,
            implementation=implementation,
            aggregator=aggregator,
            **kwargs
        )

        # By default, higher score is better
        if higher_is_better is None:
            self.higher_is_better = True
        else:
            self.higher_is_better = higher_is_better

        if feedback_definition_id is None:
            if implementation is not None:
                feedback_definition_id = obj_id_of_obj(
                    self.model_dump(), prefix="feedback_definition"
                )
            else:
                feedback_definition_id = "anonymous_feedback_definition"

        self.feedback_definition_id = feedback_definition_id

    def __hash__(self):
        return hash(self.feedback_definition_id)

    @property
    def name(self) -> str:
        """Name of the feedback function.
        
        Derived from the name of the serialized implementation function if name
        was not provided.
        """

        if self.supplied_name is not None:
            return self.supplied_name

        if self.implementation is None:
            raise RuntimeError("This feedback function has no implementation.")

        return self.implementation.name


# App related:


class FeedbackMode(str, Enum):
    NONE = "none"
    """No evaluation will happen even if feedback functions are specified."""

    WITH_APP = "with_app"
    """Try to run feedback functions immediately and before app returns a
    record."""

    WITH_APP_THREAD = "with_app_thread"
    """Try to run feedback functions in the same process as the app but after
    it produces a record."""

    DEFERRED = "deferred"
    """Evaluate later via the process started by
    `tru.start_deferred_feedback_evaluator`."""


class AppDefinition(pyschema.WithClassInfo, serial.SerialModel):
    """Serialized fields of an app here whereas [App][trulens_eval.app.App]
    contains non-serialized fields."""

    app_id: AppID  # str
    """Unique identifier for this app."""

    tags: Tags  # str
    """Tags for the app."""

    metadata: Metadata  # dict  # TODO: rename to meta for consistency with other metas
    """Metadata for the app."""

    feedback_definitions: Sequence[FeedbackDefinition] = []
    """Feedback functions to evaluate on each record."""

    feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD
    """How to evaluate feedback functions upon producing a record."""

    root_class: pyschema.Class
    """Class of the main instrumented object.
    
    Ideally this would be a [ClassVar][] but since we want to check this without
    instantiating the subclass of
    [AppDefinition][trulens_eval.schema.AppDefinition] that would define it, we
    cannot use [ClassVar][].
    """

    root_callable: ClassVar[pyschema.FunctionOrMethod]
    """App's main method. 
    
    This is to be filled in by subclass.
    """

    app: serial.JSONized[AppDefinition]
    """Wrapped app in jsonized form."""

    initial_app_loader_dump: Optional[serial.SerialBytes] = None
    """EXPERIMENTAL: serialization of a function that loads an app.

    Dump is of the initial app state before any invocations. This can be used to
    create a new session.
    """

    app_extra_json: serial.JSON
    """Info to store about the app and to display in dashboard. 
    
    This can be used even if app itself cannot be serialized. `app_extra_json`,
    then, can stand in place for whatever data the user might want to keep track
    of about the app.
    """

    def __init__(
        self,
        app_id: Optional[AppID] = None,
        tags: Optional[Tags] = None,
        metadata: Optional[Metadata] = None,
        feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD,
        app_extra_json: serial.JSON = None,
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
                    self.initial_app_loader_dump = serial.SerialBytes(data=dump)

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
        app_definition_json: serial.JSON, app: Any
    ) -> AppDefinition:
        # initial_app_loader: Optional[Callable] = None) -> 'AppDefinition':
        """EXPERIMENTAL: Instantiate the given `app` with the given state
        `app_definition_json`.
        
        Args:
            app_definition_json: The json serialized app.

            app: The app to continue the session with.
        
        Returns:
            A new `AppDefinition` instance with the given `app` and the given
                `app_definition_json` state.
        """

        app_definition_json['app'] = app

        cls = pyschema.WithClassInfo.get_class(app_definition_json)

        return cls(**app_definition_json)

    @staticmethod
    def new_session(
        app_definition_json: serial.JSON,
        initial_app_loader: Optional[Callable] = None
    ) -> AppDefinition:
        """EXPERIMENTAL: Create an app instance at the start of a session.
        
        Create a copy of the json serialized app with the enclosed app being
        initialized to its initial state before any records are produced (i.e.
        blank memory).
        """

        serial_bytes_json: Optional[serial.JSON] = app_definition_json[
            'initial_app_loader_dump']

        if initial_app_loader is None:
            assert serial_bytes_json is not None, "Cannot create new session without `initial_app_loader`."

            serial_bytes = serial.SerialBytes.model_validate(serial_bytes_json)

            app = dill.loads(serial_bytes.data)()

        else:
            app = initial_app_loader()
            data = dill.dumps(initial_app_loader, recurse=True)
            serial_bytes = serial.SerialBytes(data=data)
            serial_bytes_json = serial_bytes.model_dump()

        app_definition_json['app'] = app
        app_definition_json['initial_app_loader_dump'] = serial_bytes_json

        cls: Type[App] = pyschema.WithClassInfo.get_class(app_definition_json)

        return cls.model_validate_json(app_definition_json)

    def jsonify_extra(self, content):
        # Called by jsonify for us to add any data we might want to add to the
        # serialization of `app`.
        if self.app_extra_json is not None:
            content['app'].update(self.app_extra_json)

        return content

    @staticmethod
    def get_loadable_apps():
        """EXPERIMENTAL: Gets a list of all of the loadable apps.
        
        This is those that have `initial_app_loader_dump` set.
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

        from trulens_eval import app
        if isinstance(self, app.App):
            return jsonify(self, instrument=self.instrument)
        else:
            return jsonify(self)

    @classmethod
    def select_inputs(cls) -> serial.Lens:
        """
        Get the path to the main app's call inputs.
        """

        return getattr(
            Select.RecordCalls,
            cls.root_callable.default_factory().name
        ).args

    @classmethod
    def select_outputs(cls) -> serial.Lens:
        """
        Get the path to the main app's call outputs.
        """

        return getattr(
            Select.RecordCalls,
            cls.root_callable.default_factory().name
        ).rets

# HACK013: Need these if using __future__.annotations .
RecordAppCallMethod.model_rebuild()
Cost.model_rebuild()
Perf.model_rebuild()
Record.model_rebuild()
RecordAppCall.model_rebuild()
FeedbackResult.model_rebuild()
FeedbackCall.model_rebuild()
FeedbackDefinition.model_rebuild()
AppDefinition.model_rebuild()

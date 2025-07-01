"""Serializable record-related classes."""

from __future__ import annotations

from concurrent.futures import as_completed
import datetime
import logging
from typing import (
    ClassVar,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from munch import Munch as Bunch
import pydantic
from trulens.core._utils.pycompat import Future  # import style exception
from trulens.core.schema import base as base_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import select as select_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import threading as threading_utils

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RecordAppCallMethod(serial_utils.SerialModel):
    """Method information for the stacks inside `RecordAppCall`."""

    def __str__(self):
        return f"{self.path}.{self.method.name}"

    path: serial_utils.Lens
    """Path to the method in the app's structure."""

    method: pyschema_utils.Method
    """The method that was called."""


class RecordAppCall(serial_utils.SerialModel):
    """Info regarding each instrumented method call."""

    def __str__(self):
        stack = " -> ".join(map(str, self.stack))
        return f"RecordAppCall: {stack}"

    call_id: types_schema.CallID = pydantic.Field(
        default_factory=types_schema.new_call_id
    )
    """Unique identifier for this call.

    This is shared across different instances of
    [RecordAppCall][trulens.core.schema.record.RecordAppCall] if they refer to
    the same Python method call. This may happen if multiple recorders capture
    the call in which case they will each have a different
    [RecordAppCall][trulens.core.schema.record.RecordAppCall] but the
    [call_id][trulens.core.schema.record.RecordAppCall.call_id] will be the
    same.
    """

    stack: List[RecordAppCallMethod]
    """Call stack but only containing paths of instrumented apps/other objects."""

    args: serial_utils.JSON
    """Arguments to the instrumented method."""

    rets: Optional[serial_utils.JSON] = None
    """Returns of the instrumented method if successful.

    Sometimes this is a dict, sometimes a sequence, and sometimes a base value.
    """

    error: Optional[str] = None
    """Error message if call raised exception."""

    perf: Optional[base_schema.Perf] = None
    """Timestamps tracking entrance and exit of the instrumented method."""

    pid: int
    """Process id."""

    tid: int
    """Thread id."""

    @property
    def top(self) -> RecordAppCallMethod:
        """The top of the stack."""

        return self.stack[-1]

    @property
    def method(self) -> pyschema_utils.Method:
        """The method at the top of the stack."""

        return self.top.method


class Record(serial_utils.SerialModel, Hashable):
    """The record of a single main method call.

    Note:
        This class will be renamed to `Trace` in the future.
    """

    def __str__(self):
        ret = f"Record({self.record_id}) with {len(self.calls)} calls:\n"
        for call in self.calls:
            ret += "  " + str(call) + "\n"

        return ret

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

    record_id: types_schema.RecordID
    """Unique identifier for this record."""

    app_id: types_schema.AppID
    """The app that produced this record."""

    cost: Optional[base_schema.Cost] = None
    """Costs associated with the record."""

    perf: Optional[base_schema.Perf] = None
    """Performance information."""

    ts: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now
    )
    """Timestamp of last update.

    This is usually set whenever a record is changed in any way."""

    tags: Optional[str] = ""
    """Tags for the record."""

    meta: Optional[serial_utils.JSON] = None
    """Metadata for the record."""

    main_input: Optional[serial_utils.JSON] = None
    """The app's main input."""

    main_output: Optional[serial_utils.JSON] = None  # if no error
    """The app's main output if there was no error."""

    main_error: Optional[serial_utils.JSON] = None  # if error
    """The app's main error if there was an error."""

    calls: List[RecordAppCall] = []
    """The collection of calls recorded.

    Note that these can be converted into a json structure with the same paths
    as the app that generated this record via `layout_calls_as_app`.

    Invariant: calls are ordered by `.perf.end_time`.
    """

    feedback_and_future_results: Optional[
        List[
            Tuple[
                feedback_schema.FeedbackDefinition,
                Future[feedback_schema.FeedbackResult],
            ]
        ]
    ] = pydantic.Field(None, exclude=True)
    """Map of feedbacks to the futures for of their results.

    These are only filled for records that were just produced. This will not
    be filled in when read from database. Also, will not fill in when using
    `FeedbackMode.DEFERRED`.
    """

    feedback_results: Optional[List[Future[feedback_schema.FeedbackResult]]] = (
        pydantic.Field(None, exclude=True)
    )
    """Only the futures part of the above for backwards compatibility."""

    @property
    def feedback_results_as_completed(
        self,
    ) -> Iterable[feedback_schema.FeedbackResult]:
        """Generate feedback results as they are completed.

        Wraps
        [feedback_results][trulens.core.schema.record.Record.feedback_results]
        in [as_completed][concurrent.futures.as_completed].
        """

        if self.feedback_results is None:
            return

        for f in as_completed(self.feedback_results):
            yield f.result()

    def __init__(
        self,
        record_id: Optional[types_schema.RecordID] = None,
        calls: Optional[List[RecordAppCall]] = None,
        **kwargs,
    ):
        super(serial_utils.SerialModel, self).__init__(
            record_id="temporary", calls=calls, **kwargs
        )

        # Resetting the calls in sorted order here. Note that we are calling
        # init with calls above to make sure they get converted to RecordAppCall
        # from dicts. We then sort those in the next step:
        if calls is not None:
            self.calls = sorted(
                self.calls,
                key=lambda call: call.perf.end_time
                if call.perf is not None
                else datetime.datetime.max,
            )

        if record_id is None:
            record_id = json_utils.obj_id_of_obj(
                json_utils.jsonify(self), prefix="record"
            )

        self.record_id = record_id

    def __hash__(self):
        return hash(self.record_id)

    def wait_for_feedback_results(
        self, feedback_timeout: Optional[float] = None
    ) -> Dict[
        feedback_schema.FeedbackDefinition,
        feedback_schema.FeedbackResult,
    ]:
        """Wait for feedback results to finish.

        Args:
            feedback_timeout: Timeout in seconds for each feedback function. If
                not given, will use the default timeout
                `trulens.core.utils.threading.TP.DEBUG_TIMEOUT`.

        Returns:
            A mapping of feedback functions to their results.
        """
        if feedback_timeout is None:
            feedback_timeout = threading_utils.TP.DEBUG_TIMEOUT

        if self.feedback_and_future_results is None:
            return {}

        ret = {}

        for feedback_def, future_result in self.feedback_and_future_results:
            try:
                feedback_result = future_result.result(timeout=feedback_timeout)
                ret[feedback_def] = feedback_result
            except TimeoutError:
                logger.error(
                    "Timeout waiting for feedback result for %s.",
                    feedback_def.name,
                )

        return ret

    def get(self, path: serial_utils.Lens) -> Optional[T]:
        """Get a value from the record using a path.

        Args:
            path: Path to the value.
        """

        layout = self.layout_calls_as_app()

        if len(path.path) == 0:
            return layout

        if path.path[0] == select_schema.Select.App.path[0]:
            raise ValueError("This path references an app, not a record.")
        elif path.path[0] == select_schema.Select.Record.path[0]:
            path = serial_utils.Lens(path.path[1:])

        return path.get_sole_item(obj=layout)

    def layout_calls_as_app(self) -> Bunch:
        """Layout the calls in this record into the structure that follows that of
        the app that created this record.

        This uses the paths stored in each
        [RecordAppCall][trulens.core.schema.record.RecordAppCall] which are paths into
        the app.

        Note: We cannot create a validated
        [AppDefinition][trulens.core.schema.app.AppDefinition] class (or subclass)
        object here as the layout of records differ in these ways:

        - Records do not include anything that is not an instrumented method
          hence have most of the structure of a app missing.

        - Records have RecordAppCall as their leafs where method definitions
          would be in the AppDefinition structure.
        """

        ret = Bunch(**self.model_dump())

        for call in self.calls:
            # Info about the method call is at the top of the stack
            frame_info = call.top

            # Adds another attribute to path, from method name:
            path = frame_info.path + serial_utils.GetItemOrAttribute(
                item_or_attribute=frame_info.method.name
            )

            if path.exists(obj=ret):
                existing = path.get_sole_item(obj=ret)
                ret = path.set(obj=ret, val=existing + [call])
            else:
                ret = path.set(obj=ret, val=[call])

        ret.spans = {}
        return ret


# HACK013: Need these if using __future__.annotations .
RecordAppCallMethod.model_rebuild()
Record.model_rebuild()
RecordAppCall.model_rebuild()

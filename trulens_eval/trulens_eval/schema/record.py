"""Serializable record-related classes."""

from __future__ import annotations

import datetime
import logging
from typing import ClassVar, Dict, Hashable, List, Optional, Tuple, TypeVar

from munch import Munch as Bunch
import pydantic

from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.schema import types as mod_types_schema
from trulens_eval.utils import pyschema
from trulens_eval.utils import serial
from trulens_eval.utils import threading as mod_threading_utils
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.json import obj_id_of_obj
from trulens_eval.utils.python import Future

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RecordAppCallMethod(serial.SerialModel):
    """Method information for the stacks inside `RecordAppCall`."""

    path: serial.Lens
    """Path to the method in the app's structure."""

    method: pyschema.Method
    """The method that was called."""


class RecordAppCall(serial.SerialModel):
    """Info regarding each instrumented method call."""

    call_id: mod_types_schema.CallID = pydantic.Field(
        default_factory=mod_types_schema.new_call_id
    )
    """Unique identifier for this call.
    
    This is shared across different instances of
    [RecordAppCall][trulens_eval.schema.record.RecordAppCall] if they refer to
    the same python method call. This may happen if multiple recorders capture
    the call in which case they will each have a different
    [RecordAppCall][trulens_eval.schema.record.RecordAppCall] but the
    [call_id][trulens_eval.schema.record.RecordAppCall.call_id] will be the
    same.
    """

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

    perf: Optional[mod_base_schema.Perf] = None
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
    """The record of a single main method call.

    Note:
        This class will be renamed to `Trace` in the future.
    """

    model_config: ClassVar[dict] = {
        # for `Future[FeedbackResult]`
        'arbitrary_types_allowed': True
    }

    record_id: mod_types_schema.RecordID
    """Unique identifier for this record."""

    app_id: mod_types_schema.AppID
    """The app that produced this record."""

    cost: Optional[mod_base_schema.Cost] = None
    """Costs associated with the record."""

    perf: Optional[mod_base_schema.Perf] = None
    """Performance information."""

    ts: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now
    )
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

    feedback_and_future_results: Optional[List[
        Tuple[mod_feedback_schema.FeedbackDefinition,
              Future[mod_feedback_schema.FeedbackResult]]]] = pydantic.Field(
                  None, exclude=True
              )
    """Map of feedbacks to the futures for of their results.
     
    These are only filled for records that were just produced. This will not
    be filled in when read from database. Also, will not fill in when using
    `FeedbackMode.DEFERRED`.
    """

    feedback_results: Optional[List[Future[mod_feedback_schema.FeedbackResult]]] = \
        pydantic.Field(None, exclude=True)
    """Only the futures part of the above for backwards compatibility."""

    def __init__(
        self, record_id: Optional[mod_types_schema.RecordID] = None, **kwargs
    ):
        super().__init__(record_id="temporary", **kwargs)

        if record_id is None:
            record_id = obj_id_of_obj(jsonify(self), prefix="record")

        self.record_id = record_id

    def __hash__(self):
        return hash(self.record_id)

    def wait_for_feedback_results(
        self,
        feedback_timeout: Optional[float] = None
    ) -> Dict[mod_feedback_schema.FeedbackDefinition,
              mod_feedback_schema.FeedbackResult]:
        """Wait for feedback results to finish.

        Args:
            feedback_timeout: Timeout in seconds for each feedback function. If
                not given, will use the default timeout
                `trulens_eval.utils.threading.TP.DEBUG_TIMEOUT`. 

        Returns:
            A mapping of feedback functions to their results.
        """

        if feedback_timeout is None:
            feedback_timeout = mod_threading_utils.TP.DEBUG_TIMEOUT

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
                    feedback_def.name
                )

        return ret

    def layout_calls_as_app(self) -> Bunch:
        """Layout the calls in this record into the structure that follows that of
        the app that created this record.
        
        This uses the paths stored in each
        [RecordAppCall][trulens_eval.schema.record.RecordAppCall] which are paths into
        the app.

        Note: We cannot create a validated
        [AppDefinition][trulens_eval.schema.app.AppDefinition] class (or subclass)
        object here as the layout of records differ in these ways:
        
        - Records do not include anything that is not an instrumented method
          hence have most of the structure of a app missing.
        
        - Records have RecordAppCall as their leafs where method definitions
          would be in the AppDefinition structure.
        """

        ret = Bunch(**self.model_dump())

        for call in self.calls:
            # Info about the method call is at the top of the stack
            frame_info = call.top()

            # Adds another attribute to path, from method name:
            path = frame_info.path._append(
                serial.GetItemOrAttribute(
                    item_or_attribute=frame_info.method.name
                )
            )

            if path.exists(obj=ret):
                existing = path.get_sole_item(obj=ret)
                ret = path.set(obj=ret, val=existing + [call])
            else:
                ret = path.set(obj=ret, val=[call])

        return ret


# HACK013: Need these if using __future__.annotations .
RecordAppCallMethod.model_rebuild()
Record.model_rebuild()
RecordAppCall.model_rebuild()

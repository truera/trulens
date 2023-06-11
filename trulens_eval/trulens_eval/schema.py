"""
Serializable objects and their schemas.
"""

import abc
from datetime import datetime
from datetime import timedelta
from enum import Enum
import importlib
import json
from types import ModuleType
from typing import (
    Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar, Union
)

from munch import Munch as Bunch
import pydantic
from trulens_eval.util import all_queries
from trulens_eval.util import WithClassInfo
from trulens_eval.util import Function, Method

from trulens_eval.util import GetItemOrAttribute
from trulens_eval.util import JSON
from trulens_eval.util import json_default
from trulens_eval.util import json_str_of_obj
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.util import obj_id_of_obj
from trulens_eval.util import SerialModel

T = TypeVar("T")

# Identifier types.

RecordID = str
ChainID = str
FeedbackDefinitionID = str
FeedbackResultID = str

# Serialization of python objects/methods. Not using pickling here so we can
# inspect the contents a little better before unserializaing.

# Record related:


class RecordChainCallMethod(SerialModel):
    path: JSONPath
    method: Method


class Cost(SerialModel):
    n_tokens: Optional[int] = None
    cost: Optional[float] = None


class Latency(SerialModel):
    latency: Optional[float] = None


class RecordChainCall(SerialModel):
    """
    Info regarding each instrumented method call is put into this container.
    """

    # Call stack but only containing paths of instrumented chains/other objects.
    chain_stack: Sequence[RecordChainCallMethod]

    # Arguments to the instrumented method.
    args: JSON

    # Returns of the instrumented method if successful. Sometimes this is a
    # dict, sometimes a sequence, and sometimes a base value.
    rets: Optional[Any] = None

    # Error message if call raised exception.
    error: Optional[str] = None

    # Timestamps tracking entrance and exit of the instrumented method.
    start_time: datetime
    end_time: datetime
    latency: timedelta

    # Process id.
    pid: int

    # Thread id.
    tid: int

    def top(self):
        return self.chain_stack[-1]

    def method(self):
        return self.top().method


class Record(SerialModel):
    record_id: RecordID
    chain_id: ChainID

    cost: Cost = pydantic.Field(default_factory=Cost)
    latency: Latency = pydantic.Field(default_factory=Latency)

    ts: datetime = pydantic.Field(default_factory=lambda: datetime.now())

    tags: str = ""

    main_input: Optional[str]
    main_output: Optional[str]  # if no error
    main_error: Optional[str]  # if error

    # The collection of calls recorded. Note that these can be converted into a
    # json structure with the same paths as the chain that generated this record
    # via `layout_calls_as_chain`.
    calls: Sequence[RecordChainCall] = []

    def __init__(self, record_id: Optional[RecordID] = None, **kwargs):
        super().__init__(record_id="temporay", **kwargs)

        if record_id is None:
            record_id = obj_id_of_obj(self.dict(), prefix="record")

        self.record_id = record_id

    # TODO: typing
    def layout_calls_as_chain(self) -> Any:
        """
        Layout the calls in this record into the structure that follows that of
        the chain that created this record. This uses the paths stored in each
        `RecordChainCall` which are paths into the chain.

        Note: We cannot create a validated schema.py:Chain class (or subclass)
        object here as the layout of records differ in these ways:

            - Records do not include anything that is not an instrumented method
              hence have most of the structure of a chain missing.
        
            - Records have RecordChainCall as their leafs where method
              definitions would be in the Chain structure.
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

    # Instance for constructing queries for record json like `Record.chain.llm`.
    Record = Query().__record__

    # Instance for constructing queries for chain json.
    Chain = Query().__chain__

    # A Chain's main input and main output.
    # TODO: Chain input/output generalization.
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

    chain_id: ChainID

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
        - selectors: Optional[Dict[str, Selection]] -- mapping of implementation
          argument names to where to get them from a record.

        - feedback_definition_id: Optional[str] - unique identifier.

        - implementation:

        - aggregator:
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

# Model related:

class FeedbackMode(str, Enum):
    # No evaluation will happen even if feedback functions are specified.
    NONE = "none"

    # Try to run feedback functions immediately and before chain returns a
    # record.
    # TODO: rename
    WITH_CHAIN = "with_chain"

    # Try to run feedback functions in the same process as the chain but after
    # it produces a record.
    # TODO: rename
    WITH_CHAIN_THREAD = "with_chain_thread"

    # Evaluate later via the process started by
    # `tru.start_deferred_feedback_evaluator`.
    DEFERRED = "deferred"

class Model(SerialModel, WithClassInfo):
    # Serialized fields here whereas tru_model.py:TruMode contains
    # non-serialized fields.

    class Config:
        arbitrary_types_allowed = True

    # TODO: rename to model_id
    chain_id: ChainID

    # Feedback functions to evaluate on each record. Unlike the above, these are
    # meant to be serialized.
    feedback_definitions: Sequence[FeedbackDefinition] = []

    # NOTE: Custom feedback functions cannot be run deferred and will be run as
    # if "withchainthread" was set.
    feedback_mode: FeedbackMode = FeedbackMode.WITH_CHAIN_THREAD

    def __init__(
        self,
        chain_id: Optional[ChainID] = None,
        feedback_mode: FeedbackMode = FeedbackMode.WITH_CHAIN_THREAD,
        **kwargs
    ):
        
        # for us:
        kwargs['chain_id'] = "temporary"  # will be adjusted below
        kwargs['feedback_mode'] = feedback_mode

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(**kwargs)

        if chain_id is None:
            chain_id = obj_id_of_obj(obj=self.dict(), prefix="chain")

        self.chain_id = chain_id


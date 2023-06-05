"""
Serializable objects and their schemas.
"""

import abc
from datetime import datetime
from enum import Enum
import importlib
import json
from types import ModuleType
from typing import (
    Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar, Union
)

import langchain
from munch import Munch as Bunch
import pydantic
from trulens_eval.util import Function, Method, MethodIdent

from trulens_eval.util import GetItemOrAttribute, SerialModel, json_str_of_obj
from trulens_eval.util import JSON
from trulens_eval.util import json_default
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.util import obj_id_of_obj

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
    method: MethodIdent


class Cost(SerialModel):
    n_tokens: Optional[int] = None
    cost: Optional[float] = None


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

    ts: datetime = pydantic.Field(default_factory=lambda: datetime.now())

    tags: str = ""

    main_input: Optional[str]
    main_output: Optional[str] # if no error
    main_error: Optional[str] # if error

    # The collection of calls recorded. Note that these can be converted into a
    # json structure with the same paths as the chain that generated this record
    # via `layout_calls_as_chain`.
    calls: Sequence[RecordChainCall] = []

    def __init__(self,
                 record_id: Optional[RecordID] = None,

                 **kwargs):
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
                GetItemOrAttribute(item_or_attribute=frame_info.method.method_name)
            )  # adds another attribute to path, from method name
            # TODO: append if already there
            ret = path.set(obj=ret, val=call)

        return ret


# Feedback related:


class FeedbackResultStatus(Enum):
    NONE = "none"
    RUNNING = "running"
    FAILED = "failed"
    DONE = "done"


class FeedbackResult(SerialModel):
    record_id: RecordID
    chain_id: ChainID

    feedback_result_id: FeedbackResultID

    feedback_definition_id: Optional[FeedbackDefinitionID] = None

    # "last timestamp"
    last_ts: datetime = pydantic.Field(default_factory=lambda: datetime.now())

    status: FeedbackResultStatus = FeedbackResultStatus.NONE

    error: Optional[str] = None  # if there was an error

    results_json: JSON = pydantic.Field(default_factory=dict) # keeping unrestricted in type for now

    cost: Cost = pydantic.Field(default_factory=Cost)

    tags: str = ""

    def __init__(
        self,
        feedback_result_id: Optional[FeedbackResultID] = None,
        **kwargs
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

    """
    def parse_obj(o: Dict) -> 'FeedbackDefinition':
        implementation = o['implementation']
        if implementation is not None:
            o['implementation'] = FunctionOrMethod.parse_obj(implementation)

        aggregator = o['aggregator']
        if aggregator is not None:
            o['aggregator'] = FunctionOrMethod.parse_obj(aggregator)

        o['selectors'] = {k: JSONPath.parse_obj(v) for k, v in o['selectors'].items()}

        return FeedbackDefinition(**o)
    """


# Model/chain related:


class FeedbackMode(str, Enum):
    # No evaluation will happen even if feedback functions are specified.
    NONE = "none"

    # Try to run feedback functions immediately and before chain returns a
    # record.
    WITH_CHAIN = "with_chain"

    # Try to run feedback functions in the same process as the chain but after
    # it produces a record.
    WITH_CHAIN_THREAD = "with_chain_thread"

    # Evaluate later via the process started by
    # `tru.start_deferred_feedback_evaluator`.
    DEFERRED = "deferred"


class Model(SerialModel):
    """
    Base container for any model that can be instrumented with trulens.
    """

    chain_id: ChainID

    # NOTE: Custom feedback functions cannot be run deferred and will be run as
    # if "withchainthread" was set.
    feedback_mode: FeedbackMode = FeedbackMode.WITH_CHAIN_THREAD

    # Flag of whether the chain is currently recording records. This is set
    # automatically but is imperfect in threaded situations. The second check
    # for recording is based on the call stack, see _call.
    recording: bool = False

    # Feedback functions to evaluate on each record.
    feedback_definitions: Sequence[FeedbackDefinition] = []

    def __init__(
        self,
        chain_id: Optional[ChainID] = None,
        feedback_mode: FeedbackMode = FeedbackMode.WITH_CHAIN_THREAD,
        feedback_definitions: Sequence[FeedbackDefinition] = [],
        recording: bool = False,
        **kwargs
    ):

        super().__init__(
            chain_id="temporary",  # temporary for validation
            feedback_mode=feedback_mode,
            feedback_definitions=feedback_definitions,
            recording=recording,
            **kwargs
        )

        # Create a chain_id from entire serializable structure, even stuff
        # coming from subclass.
        if chain_id is None:
            chain_id = obj_id_of_obj(obj=self.dict(), prefix="chain")

        self.chain_id = chain_id

    def json(self, *args, **kwargs):
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json_str_of_obj(self.dict(), *args, **kwargs)

    def dict(self):
        # Same problem as in json.
        return jsonify(self)


class LangChainModel(langchain.chains.base.Chain, Model):
    """
    Instrumented langchain chain.
    """

    # The wrapped/instrumented chain.
    chain: langchain.chains.base.Chain

    # TODO: Consider


class LlamaIndexModel(Model):
    """
    TODO: Instrumented llama index model.
    """
    pass

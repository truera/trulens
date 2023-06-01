"""
Serializable objects and their schemas.
"""

import abc
import datetime
from enum import Enum
import json
from typing import (Any, Callable, Dict, Iterable, Optional, Sequence, Tuple,
                    TypeVar, Union)

import langchain
from munch import Munch as Bunch
import pydantic

from trulens_eval.util import GetItemOrAttribute
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


class Class(pydantic.BaseModel):
    """
    A python class by name.
    """

    class_name: str
    module_name: str

    @staticmethod
    def of_class(cls: type) -> 'Class':
        return Class(class_name=cls.__name__, module_name=cls.__module__)


# Serialization of python objects/methods:
class Obj(pydantic.BaseModel):
    """
    An object that may or may not be serializable. Do not use for base types
    that don't have a class.
    """

    cls: Class

    # From id(obj), identifiers memory location of a python object. Use this for
    # handling loops in JSON objects.
    id: int

    @staticmethod
    def of_object(obj: object, cls: Optional[type] = None) -> 'Obj':
        if cls is None:
            cls = obj.__class__

        return Obj(cls=Class.of_class(cls), id=id(obj))


class Method(pydantic.BaseModel):
    """
    A python method. A method belongs to some class in some module and must have
    a pre-bound self object. The location of the method is encoded in `obj`
    alongside self.
    """

    obj: Obj
    method_name: str

    @staticmethod
    def of_method(
        meth: Callable,
        cls: Optional[type] = None,
        obj: Optional[object] = None
    ) -> 'Method':
        if obj is None:
            assert hasattr(
                meth, "__self__"
            ), f"Expected a method (maybe it is a function?): {meth}"
            obj = meth.__self__

        if cls is None:
            cls = obj.__class__

        obj_json = Obj.of_object(obj, cls=cls)

        return Method(obj=obj_json, method_name=meth.__name__)


# Record related:


class RecordChainCallMethod(pydantic.BaseModel):
    path: JSONPath
    method: Method


class RecordCost(pydantic.BaseModel):
    n_tokens: Optional[int]
    cost: Optional[float]


class RecordChainCall(pydantic.BaseModel):
    """
    Info regarding each instrumented method call is put into this container.
    """

    # Call stack but only containing paths of instrumented chains/other objects.
    chain_stack: Sequence[RecordChainCallMethod]

    # Arguments to the instrumented method.
    args: JSON

    # Returns of the instrumented method if successful.
    rets: Optional[JSON] = None

    # Error message if call raised exception.
    error: Optional[str] = None

    # Timestamps tracking entrance and exit of the instrumented method.
    start_time: datetime.datetime
    end_time: datetime.datetime

    # Process id.
    pid: int

    # Thread id.
    tid: int

    def top(self):
        return self.chain_stack[-1]

    def method(self):
        return self.top().method


class Record(pydantic.BaseModel):
    record_id: RecordID
    chain_id: ChainID

    cost: RecordCost

    main_input: str
    main_output: Optional[str]  # if no error
    main_error: Optional[str]  # if error

    # The collection of calls recorded. Note that these can be converted into a
    # json structure with the same paths as the chain that generated this record
    # via `layout_calls_as_chain`.
    calls: Sequence[RecordChainCall]

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
                GetItemOrAttribute(item=frame_info.method.method_name)
            )  # adds another attribute to path, from method name
            # TODO: append if already there
            ret = path.set(obj=ret, val=call)

        return ret


# Feedback related:


class FeedbackResult(pydantic.BaseModel):
    record_id: RecordID
    chain_id: ChainID
    feedback_result_id: FeedbackResultID
    feedback_definition_id: Optional[FeedbackDefinitionID]

    results_json: JSON

    def __init__(
        self, feedback_result_id: Optional[FeedbackResultID] = None, **kwargs
    ):
        super().__init__(feedback_result_id="temporary", **kwargs)

        if feedback_result_id is None:
            feedback_result_id = obj_id_of_obj(
                self.dict(), prefix="feedback_result"
            )

        self.feedback_result_id = feedback_result_id


Selection = Union[JSONPath, str]


class FeedbackDefinition(pydantic.BaseModel):
    # Implementation serialization info.
    implementation: Optional[Method] = None

    # Aggregator method for serialization.
    aggregator: Optional[Method] = None

    # Id, if not given, unique determined from _json below.
    feedback_definition_id: FeedbackDefinitionID

    # Selectors, pointers into Records of where to get
    # arguments for `imp`.
    selectors: Optional[Dict[str, Selection]] = None

    def __init__(
        self,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        implementation: Optional[Method] = None,
        aggregator: Optional[Method] = None,
        selectors: Dict[str, Selection] = None
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


# Model/chain related:


class FeedbackMode(Enum):
    # No evaluation will happen even if feedback functions are specified.
    NONE = 0

    # Try to run feedback functions immediately and before chain returns a
    # record.
    WITH_CHAIN = 1

    # Try to run feedback functions in the same process as the chain but after
    # it produces a record.
    WITH_CHAIN_THREAD = 2

    # Evaluate later via the process started by
    # `tru.start_deferred_feedback_evaluator`.
    DEFERRED = 3


class Model(pydantic.BaseModel):
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

    def json(self):
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json.dumps(jsonify(self.__fields__), default=json_default)

    def dict(self):
        # Same problem as in json.
        return jsonify(self.__fields__)


class LangChainModel(Model):
    """
    Instrumented langchain chain.
    """

    # The wrapped/instrumented chain.
    chain: langchain.chains.base.Chain


class LlamaIndexModel(Model):
    """
    TODO: Instrumented llama index model.
    """
    pass

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


class Module(SerialModel):
    package_name: str
    module_name: str

    def of_module(mod: ModuleType) -> 'Module':
        return Module(package_name=mod.__package__, module_name=mod.__name__)

    def of_module_name(module_name: str) -> 'Module':
        mod = importlib.import_module(module_name)
        package_name = mod.__package__
        return Module(package_name=package_name, module_name=module_name)

    def load(self) -> ModuleType:
        return importlib.import_module(
            self.module_name, package=self.package_name
        )


class Class(SerialModel):
    """
    A python class by name.
    """

    name: str

    module: Module

    @staticmethod
    def of_class(cls: type) -> 'Class':
        return Class(
            name=cls.__name__, module=Module.of_module_name(cls.__module__)
        )

    def load(self) -> type:  # class
        try:
            mod = self.module.load()
            return getattr(mod, self.name)

        except Exception as e:
            raise RuntimeError(f"Could not load class {self} because {e}.")


class Obj(SerialModel):
    """
    An object that may or may not be serializable. Do not use for base types
    that don't have a class.
    """

    cls: Class

    # From id(obj), identifiers memory location of a python object. Use this for
    # handling loops in JSON objects.
    id: int

    # For objects that can be easily reconstructed, provide their kwargs here.
    init_kwargs: Optional[Dict] = None

    @staticmethod
    def of_object(obj: object, cls: Optional[type] = None) -> 'Obj':
        if cls is None:
            cls = obj.__class__

        if isinstance(obj, pydantic.BaseModel):
            init_kwargs = obj.dict()
        else:
            init_kwargs = None

        return Obj(cls=Class.of_class(cls), id=id(obj), init_kwargs=init_kwargs)

    def load(self) -> object:
        cls = self.cls.load()

        if issubclass(cls, pydantic.BaseModel) and self.init_kwargs is not None:
            return cls(**self.init_kwargs)
        else:
            raise RuntimeError(f"Do not know how to load object {self}.")


# FunctionOrMethod = Union[Function, Method]


class FunctionOrMethod(SerialModel):  #, abc.ABC):

    @staticmethod
    def pick(**kwargs):
        if 'obj' in kwargs:
            return Method(**kwargs)
        elif 'cls' in kwargs:
            return Function(**kwargs)

    @classmethod
    def __get_validator__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, d) -> 'FunctionOrMethod':
        if isinstance(d, Function):
            return d
        elif isinstance(d, Method):
            return d
        elif isinstance(d, Dict):
            return FunctionOrMethod.pick(**d)
        else:
            raise RuntimeError(
                f"Unhandled FunctionOrMethod source of type {type(d)}."
            )

    @staticmethod
    def of_callable(c: Callable) -> 'FunctionOrMethod':
        if hasattr(c, "__self__"):
            return Method.of_method(c, obj=getattr(c, "__self__"))
        else:
            return Function.of_function(c)

    #@abc.abstractmethod
    def load(self) -> Callable:
        raise NotImplementedError()


class MethodIdent(SerialModel):
    """
    Identifier of a method (as opposed to a serialization of the method itself).
    """

    module_name: str
    class_name: str
    method_name: str

    @staticmethod
    def of_method(
        method: Callable,
        cls: Optional[type] = None,
        obj: Optional[object] = None
    ) -> 'Method':
        if obj is None:
            assert hasattr(
                method, "__self__"
            ), f"Expected a method (maybe it is a function?): {method}"
            obj = method.__self__

        if cls is None:
            cls = obj.__class__

        module_name = cls.__module__

        return MethodIdent(
            module_name=module_name,
            class_name=cls.__name__,
            method_name=method.__name__
        )


class Method(FunctionOrMethod):
    """
    A python method. A method belongs to some class in some module and must have
    a pre-bound self object. The location of the method is encoded in `obj`
    alongside self.
    """

    obj: Obj
    name: str

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

        return Method(obj=obj_json, name=meth.__name__)

    def load(self) -> Callable:
        obj = self.obj.load()
        return getattr(obj, self.name)

    # @staticmethod
    # def parse_obj(o: Dict) -> 'Method':
    #    return Method(obj=Obj.parse_obj(o['obj']), name=o['name'])


class Function(FunctionOrMethod):
    """
    A python function.
    """

    module: Module
    cls: Optional[Class]
    name: str

    @staticmethod
    def of_function(
        func: Callable,
        module: Optional[ModuleType] = None,
        cls: Optional[type] = None
    ) -> 'Function':  # actually: class

        if module is None:
            module = Module.of_module_name(func.__module__)

        if cls is not None:
            cls = Class.of_class(cls)

        return Function(cls=cls, module=module, name=func.__name__)

    def load(self) -> Callable:
        if self.cls is not None:
            cls = self.cls.load()
            return getattr(cls, self.name)
        else:
            mod = self.module.load()
            return getattr(mod, self.name)

    # @staticmethod
    # def parse_obj(o: Dict) -> 'Function':
    #    return Function(
    #        module=Module.parse_obj(o['module']),
    #        cls=Obj.parse_obj(o['cls']) if o['cls'] is not None else None,
    #        name=o['name']
    #    )


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
                GetItemOrAttribute(
                    item_or_attribute=frame_info.method.method_name
                )
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

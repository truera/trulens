"""
Serializable objects and their schemas.
"""

import abc
from typing import (Any, Callable, Dict, Iterable, Optional, Sequence, Tuple,
                    TypeVar, Union)

import pydantic

JSON_BASES = (str, int, float, type(None))
JSON_BASES_T = Union[str, int, float, type(None)]
# JSON = (List, Dict) + JSON_BASES
# JSON_T = Union[JSON_BASES_T, List, Dict]
JSON = Dict

T = TypeVar("T")


class Obj(pydantic.BaseModel):
    """
    An object that may or may not be serializable.
    """
    class_name: str
    module_name: str


    # From id(obj), identifiers memory location of a python object. Use this for handling loops in JSON objects.
    id: int 

class FeedbackImplementation(pydantic.BaseModel):
    cls: Obj

    kwargs: Dict[str, JSON]

class Step(pydantic.BaseModel):
    """
    A step in a selection path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #@abc.abstractmethod
    #def __call__(self, obj: Any) -> Iterable[Any]:
    #    """
    #    Get the element of `obj`, indexed by `self`.
    #    """
#
 #       raise NotImplementedError


class GetAttribute(Step):
    attribute: str

    def __call__(self, obj: Any) -> Iterable[Any]:
        if hasattr(obj, self.attribute):
            yield getattr(obj, self.attribute)
        else:
            raise ValueError(
                f"Object does not have attribute: {self.attribute}"
            )

    def __repr__(self):
        return f".{self.attribute}"


class GetIndex(Step):
    index: int

    def __call__(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            if len(obj) > self.index:
                yield obj[self.index]
            else:
                raise IndexError(f"Index out of bounds: {self.index}")
        else:
            raise ValueError("Object is not a sequence.")

    def __repr__(self):
        return f"[{self.index}]"


class GetItem(Step):
    item: str

    def __call__(self, obj: Dict[str, T]) -> Iterable[T]:
        if isinstance(obj, Dict):
            if self.item in obj:
                yield obj[self.item]
            else:
                raise KeyError(f"Key not in dictionary: {self.item}")
        else:
            raise ValueError("Object is not a dictionary.")

    def __repr__(self):
        return f"[{self.item}]"


class GetSlice(Step):
    slice: Tuple[int, int, int]

    def __call__(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            lower, upper, step = slice(**self.slice).indices(len(obj))
            for i in range(lower, upper, step):
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    def __repr__(self):
        return f"[{self.slice[0]}:{self.slice[1]}:{self.slice[2]}]"



class GetIndices(Step):
    indices: Sequence[int]

    def __call__(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            for i in self.indices:
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    def __repr__(self):
        return f"[{','.join(map(str, self.indices))}]"


class GetItems(Step):
    items: Sequence[str]

    def __call__(self, obj: Dict[str, T]) -> Iterable[T]:
        if isinstance(obj, Dict):
            for i in self.items:
                yield obj[i]
        else:
            raise ValueError("Object is not a dictionary.")

    def __repr__(self):
        return f"[{','.join(self.indices)}]"


class JSONPath(pydantic.BaseModel):
    path: Tuple[Step]


class Chain(pydantic.BaseModel):
    chain_id: str
    chain: JSON  # langchain structure


class RecordChainCallFrame(pydantic.BaseModel):
    path: JSONPath
    method_name: str
    class_name: str
    module_name: str


class RecordCost(pydantic.BaseModel):
    n_tokens: Optional[int]
    cost: Optional[float]


class RecordChainCall(pydantic.BaseModel):
    """
    Info regarding each instrumented method call is put into this container.
    """

    # Call stack but only containing paths of instrumented chains/other objects.
    chain_stack: Sequence[RecordChainCallFrame]

    # Arguments to the instrumented method.
    args: Dict

    # Returns of the instrumented method.
    rets: Dict

    # Error message if call raised exception.
    error: Optional[str]

    # Timestamps tracking entrance and exit of the instrumented method.
    start_time: int
    end_int: int

    # Process id.
    pid: int

    # Thread id.
    tid: int


class Record(pydantic.BaseModel):
    record_id: str
    chain_id: str

    cost: RecordCost

    total_tokens: int
    total_cost: float

    calls: Sequence[
        RecordChainCall
    ]  # not the actual chain, but rather json structure that mirrors the chain structure


class FeedbackResult(pydantic.BaseModel):
    record_id: str
    chain_id: str
    feedback_id: Optional[str]

    results_json: JSON


Selection = Union[JSONPath, str]


class FeedbackDefinition(pydantic.BaseModel):
    # Implementation serialization info.
    imp_json: Optional[JSON] = pydantic.Field(exclude=True)

    # Id, if not given, unique determined from _json below.
    feedback_id: Optional[str] = None

    # Selectors, pointers into Records of where to get
    # arguments for `imp`.
    selectors: Optional[Dict[str, Selection]] = None

    # TODO: remove
    # JSON version of this object.
    feedback_json: Optional[JSON] = pydantic.Field(exclude=True)
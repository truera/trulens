"""
Utilities.

Do not import anything from trulens_eval here.
"""

from __future__ import annotations

import abc
from enum import Enum
from inspect import stack
import itertools
import json
import logging
from multiprocessing.context import TimeoutError
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import ThreadPool
from pathlib import Path
from queue import Queue
from threading import Thread
from time import sleep
from typing import (
    Any, Callable, Dict, Hashable, Iterable, Iterator, List, Optional, Sequence,
    Set, Tuple, TypeVar, Union
)

from merkle_json import MerkleJson
from munch import Munch as Bunch
import pandas as pd
import pydantic

logger = logging.getLogger(__name__)

T = TypeVar("T")

UNICODE_CHECK = "✅"
UNCIODE_YIELD = "⚡"

# Collection utilities


def first(seq: Sequence[T]) -> T:
    return seq[0]


def second(seq: Sequence[T]) -> T:
    return seq[1]


def third(seq: Sequence[T]) -> T:
    return seq[2]


# Generator utils


def iterable_peek(it: Iterable[T]) -> Tuple[T, Iterable[T]]:
    iterator = iter(it)
    item = next(iterator)
    return item, itertools.chain([item], iterator)


# JSON utilities

JSON_BASES = (str, int, float, type(None))
JSON_BASES_T = Union[str, int, float, type(None)]
# JSON = (List, Dict) + JSON_BASES
# JSON_T = Union[JSON_BASES_T, List, Dict]
JSON = Union[JSON_BASES_T, Dict[str, Any]]
# want: Union[JSON_BASES_T, Dict[str, JSON]] but this will result in loop at some point

mj = MerkleJson()


def is_empty(obj):
    try:
        return len(obj) == 0
    except Exception:
        return False


def is_noserio(obj):
    """
    Determines whether the given json object represents some non-serializable
    object. See `noserio`.
    """
    return isinstance(obj, dict) and "_NON_SERIALIZED_OBJECT" in obj


def noserio(obj, **extra: Dict) -> dict:
    """
    Create a json structure to represent a non-serializable object. Any
    additional keyword arguments are included.
    """

    inner = {
        "id": id(obj),
        "class": obj.__class__.__name__,
        "module": obj.__class__.__module__,
        "bases": list(map(lambda b: b.__name__, obj.__class__.__bases__))
    }
    inner.update(extra)

    return {'_NON_SERIALIZED_OBJECT': inner}


def obj_id_of_obj(obj: dict, prefix="obj"):
    """
    Create an id from a json-able structure/definition. Should produce the same
    name if definition stays the same.
    """

    return f"{prefix}_hash_{mj.hash(obj)}"


def json_str_of_obj(obj: Any, *args, **kwargs) -> str:
    """
    Encode the given json object as a string.
    """

    if isinstance(obj, pydantic.BaseModel):
        kwargs['encoder'] = json_default
        return obj.json(*args, **kwargs)

    return json.dumps(obj, default=json_default)


def json_default(obj: Any) -> str:
    """
    Produce a representation of an object which cannot be json-serialized.
    """

    # obj = jsonify(obj)

    # Try the encoders included with pydantic first (should handle things like
    # Datetime):
    try:
        return pydantic.json.pydantic_encoder(obj)
    except:
        pass

    #if isinstance(obj, Enum):
    #    return obj.name

    #if isinstance(obj, dict):
    #    return

    #if isinstance(obj, pydantic.BaseModel):
    #    try:
    #        return json.dumps(obj.json())
    #    except Exception as e:
    #        return noserio(obj, exception=e)

    # Intentionally not including much in this indicator to make sure the model
    # hashing procedure does not get randomized due to something here.

    return noserio(obj)


def jsonify(obj: Any, dicted=None) -> JSON:
    """
    Convert the given object into types that can be serialized in json.
    """

    dicted = dicted or dict()

    if isinstance(obj, JSON_BASES):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if type(obj) in pydantic.json.ENCODERS_BY_TYPE:
        return obj
    # if isinstance(obj, Enum):
    #    return str(obj)

    if id(obj) in dicted:
        return {'_CIRCULAR_REFERENCE': id(obj)}

    new_dicted = {k: v for k, v in dicted.items()}

    if isinstance(obj, Enum):
        return obj.name

    if isinstance(obj, Dict):
        temp = {}
        new_dicted[id(obj)] = temp
        temp.update({k: jsonify(v, dicted=new_dicted) for k, v in obj.items()})
        return temp

    elif isinstance(obj, Sequence):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (jsonify(v, dicted=new_dicted) for v in obj):
            temp.append(x)
        return temp

    elif isinstance(obj, Set):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (jsonify(v, dicted=new_dicted) for v in obj):
            temp.append(x)
        return temp

    elif isinstance(obj, pydantic.BaseModel):
        temp = {}
        new_dicted[id(obj)] = temp
        temp.update(
            {
                k: jsonify(getattr(obj, k), dicted=new_dicted)
                for k in obj.__fields__
            }
        )
        return temp

    else:
        logger.debug(
            f"Don't know how to jsonify an object '{str(obj)[0:32]}' of type '{type(obj)}'."
        )

        return noserio(obj)


def leaf_queries(obj_json: JSON, query: JSONPath = None) -> Iterable[JSONPath]:
    """
    Get all queries for the given object that select all of its leaf values.
    """

    query = query or JSONPath()

    if isinstance(obj_json, JSON_BASES):
        yield query

    elif isinstance(obj_json, Dict):
        for k, v in obj_json.items():
            sub_query = query[k]
            for res in leaf_queries(obj_json[k], sub_query):
                yield res

    elif isinstance(obj_json, Sequence):
        for i, v in enumerate(obj_json):
            sub_query = query[i]
            for res in leaf_queries(obj_json[i], sub_query):
                yield res

    else:
        yield query


def all_queries(obj: Any, query: JSONPath = None) -> Iterable[JSONPath]:
    """
    Get all queries for the given object.
    """

    query = query or JSONPath()

    if isinstance(obj, JSON_BASES):
        yield query

    elif isinstance(obj, pydantic.BaseModel):
        yield query

        for k in obj.__fields__:
            v = getattr(obj, k)
            sub_query = query[k]
            for res in all_queries(v, sub_query):
                yield res

    elif isinstance(obj, Dict):
        yield query

        for k, v in obj.items():
            sub_query = query[k]
            for res in all_queries(obj[k], sub_query):
                yield res

    elif isinstance(obj, Sequence):
        yield query

        for i, v in enumerate(obj):
            sub_query = query[i]
            for res in all_queries(obj[i], sub_query):
                yield res

    else:
        yield query


def all_objects(obj: Any,
                query: JSONPath = None) -> Iterable[Tuple[JSONPath, Any]]:
    """
    Get all queries for the given object.
    """

    query = query or JSONPath()

    if isinstance(obj, JSON_BASES):
        yield (query, obj)

    elif isinstance(obj, pydantic.BaseModel):
        yield (query, obj)

        for k in obj.__fields__:
            v = getattr(obj, k)
            sub_query = query[k]
            for res in all_objects(v, sub_query):
                yield res

    elif isinstance(obj, Dict):
        yield (query, obj)

        for k, v in obj.items():
            sub_query = query[k]
            for res in all_objects(obj[k], sub_query):
                yield res

    elif isinstance(obj, Sequence):
        yield (query, obj)

        for i, v in enumerate(obj):
            sub_query = query[i]
            for res in all_objects(obj[i], sub_query):
                yield res

    else:
        yield (query, obj)


def leafs(obj: Any) -> Iterable[Tuple[str, Any]]:
    for q in leaf_queries(obj):
        path_str = _query_str(q)
        val = _project(q._path, obj)
        yield (path_str, val)


def matching_queries(obj: Any, match: Callable) -> Iterable[JSONPath]:
    for q in all_queries(obj):
        val = _project(q._path, obj)
        if match(q, val):
            yield q


def matching_objects(obj: Any,
                     match: Callable) -> Iterable[Tuple[JSONPath, Any]]:
    for q, val in all_objects(obj):
        if match(q, val):
            yield (q, val)


def _query_str(query: JSONPath) -> str:

    def render(ks):
        if len(ks) == 0:
            return ""

        first = ks[0]
        if len(ks) > 1:
            rest = ks[1:]
        else:
            rest = ()

        if isinstance(first, str):
            return f".{first}{render(rest)}"
        elif isinstance(first, int):
            return f"[{first}]{render(rest)}"
        else:
            RuntimeError(
                f"Don't know how to render path element {first} of type {type(first)}."
            )

    return "Record" + render(query._path)


@staticmethod
def set_in_json(query: JSONPath, in_json: JSON, val: JSON) -> JSON:
    return _set_in_json(query._path, in_json=in_json, val=val)


@staticmethod
def _set_in_json(path, in_json: JSON, val: JSON) -> JSON:
    if len(path) == 0:
        if isinstance(in_json, Dict):
            assert isinstance(val, Dict)
            in_json = {k: v for k, v in in_json.items()}
            in_json.update(val)
            return in_json

        assert in_json is None, f"Cannot set non-None json object: {in_json}"

        return val

    if len(path) == 1:
        first = path[0]
        rest = []
    else:
        first = path[0]
        rest = path[1:]

    if isinstance(first, str):
        if isinstance(in_json, Dict):
            in_json = {k: v for k, v in in_json.items()}
            if not first in in_json:
                in_json[first] = None
        elif in_json is None:
            in_json = {first: None}
        else:
            raise RuntimeError(
                f"Do not know how to set path {path} in {in_json}."
            )

        in_json[first] = _set_in_json(
            path=rest, in_json=in_json[first], val=val
        )
        return in_json

    elif isinstance(first, int):
        if isinstance(in_json, Sequence):
            # In case it is some immutable sequence. Also copy.
            in_json = list(in_json)
        elif in_json is None:
            in_json = []
        else:
            raise RuntimeError(
                f"Do not know how to set path {path} in {in_json}."
            )

        while len(in_json) <= first:
            in_json.append(None)

        in_json[first] = _set_in_json(
            path=rest, in_json=in_json[first], val=val
        )
        return in_json

    else:
        raise RuntimeError(f"Do not know how to set path {path} in {in_json}.")


# TODO: remove
def _project(path: List, obj: Any):
    if len(path) == 0:
        return obj

    first = path[0]
    if len(path) > 1:
        rest = path[1:]
    else:
        rest = ()

    if isinstance(first, str):
        if isinstance(obj, pydantic.BaseModel):
            if not hasattr(obj, first):
                logger.warn(
                    f"Cannot project {str(obj)[0:32]} with path {path} because {first} is not an attribute here."
                )
                return None
            return _project(path=rest, obj=getattr(obj, first))

        elif isinstance(obj, Dict):
            if first not in obj:
                logger.warn(
                    f"Cannot project {str(obj)[0:32]} with path {path} because {first} is not a key here."
                )
                return None
            return _project(path=rest, obj=obj[first])

        else:
            logger.warn(
                f"Cannot project {str(obj)[0:32]} with path {path} because object is not a dict or model."
            )
            return None

    elif isinstance(first, int):
        if not isinstance(obj, Sequence) or first >= len(obj):
            logger.warn(f"Cannot project {str(obj)[0:32]} with path {path}.")
            return None

        return _project(path=rest, obj=obj[first])
    else:
        raise RuntimeError(
            f"Don't know how to locate element with key of type {first}"
        )


class SerialModel(pydantic.BaseModel):
    """
    Trulens-specific additions on top of pydantic models. Includes utilities to help serialization mostly.
    """

    def update(self, **d):
        for k, v in d.items():
            setattr(self, k, v)

        return self


# JSONPath, a container for selector/accessors/setters of data stored in a json
# structure. Cannot make abstract since pydantic will try to initialize it.
class Step(SerialModel):  #, abc.ABC):
    """
    A step in a selection path.
    """

    @classmethod
    def __get_validator__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, d):
        if not isinstance(d, Dict):
            return d

        ATTRIBUTE_TYPE_MAP = {
            'item': GetItem,
            'index': GetIndex,
            'attribute': GetAttribute,
            'item_or_attribute': GetItemOrAttribute,
            'start': GetSlice,
            'stop': GetSlice,
            'step': GetSlice,
            'items': GetItems,
            'indices': GetIndices
        }

        a = next(iter(d.keys()))
        if a in ATTRIBUTE_TYPE_MAP:
            return ATTRIBUTE_TYPE_MAP[a](**d)
        else:
            raise RuntimeError(f"Don't know how to deserialize Step with {d}.")

    # @abc.abstractmethod
    def __call__(self, obj: Any) -> Iterable[Any]:
        """
        Get the element of `obj`, indexed by `self`.
        """
        raise NotImplementedError()

    # @abc.abstractmethod
    def set(self, obj: Any, val: Any) -> Any:
        """
        Set the value(s) indicated by self in `obj` to value `val`.
        """
        raise NotImplementedError()


class GetAttribute(Step):
    attribute: str

    def __hash__(self):
        return hash(self.attribute)

    def __call__(self, obj: Any) -> Iterable[Any]:
        if hasattr(obj, self.attribute):
            yield getattr(obj, self.attribute)
        else:
            raise ValueError(
                f"Object {obj} does not have attribute: {self.attribute}"
            )

    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = Bunch()

        if hasattr(obj, self.attribute):
            setattr(obj, self.attribute, val)
            return obj
        else:
            # might fail
            setattr(obj, self.attribute, val)
            return obj

    def __repr__(self):
        return f".{self.attribute}"


class GetIndex(Step):
    index: int

    def __hash__(self):
        return hash(self.index)

    def __call__(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            if len(obj) > self.index:
                yield obj[self.index]
            else:
                raise IndexError(f"Index out of bounds: {self.index}")
        else:
            raise ValueError(f"Object {obj} is not a sequence.")

    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = []

        assert isinstance(obj, Sequence), "Sequence expected."

        if self.index >= 0:
            while len(obj) <= self.index:
                obj.append(None)

        obj[self.index] = val
        return obj

    def __repr__(self):
        return f"[{self.index}]"


class GetItem(Step):
    item: str

    def __hash__(self):
        return hash(self.item)

    def __call__(self, obj: Dict[str, T]) -> Iterable[T]:
        if isinstance(obj, Dict):
            if self.item in obj:
                yield obj[self.item]
            else:
                raise KeyError(f"Key not in dictionary: {self.item}")
        else:
            raise ValueError(f"Object {obj} is not a dictionary.")

    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = dict()

        assert isinstance(obj, Dict), "Dictionary expected."

        obj[self.item] = val
        return obj

    def __repr__(self):
        return f"[{repr(self.item)}]"


class GetItemOrAttribute(Step):
    # For item/attribute agnostic addressing.

    item_or_attribute: str  # distinct from "item" for deserialization

    def __hash__(self):
        return hash(self.item)

    def __call__(self, obj: Dict[str, T]) -> Iterable[T]:
        if isinstance(obj, Dict):
            if self.item_or_attribute in obj:
                yield obj[self.item_or_attribute]
            else:
                raise KeyError(
                    f"Key not in dictionary: {self.item_or_attribute}"
                )
        else:
            if hasattr(obj, self.item_or_attribute):
                yield getattr(obj, self.item_or_attribute)
            else:
                raise ValueError(
                    f"Object {obj} does not have item or attribute {self.item_or_attribute}."
                )

    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = dict()

        if isinstance(obj, Dict):
            obj[self.item_or_attribute] = val
        else:
            setattr(obj, self.item_or_attribute)

        return obj

    def __repr__(self):
        return f".{self.item_or_attribute}"


class GetSlice(Step):
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]

    def __hash__(self):
        return hash((self.start, self.stop, self.step))

    def __call__(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            lower, upper, step = slice(self.start, self.stop,
                                       self.step).indices(len(obj))
            for i in range(lower, upper, step):
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = []

        assert isinstance(obj, Sequence), "Sequence expected."

        lower, upper, step = slice(self.start, self.stop,
                                   self.step).indices(len(obj))

        for i in range(lower, upper, step):
            obj[i] = val

        return obj

    def __repr__(self):
        pieces = ":".join(
            [
                "" if p is None else str(p)
                for p in (self.start, self.stop, self.step)
            ]
        )
        if pieces == "::":
            pieces = ":"

        return f"[{pieces}]"


class GetIndices(Step):
    indices: Sequence[int]

    def __hash__(self):
        return hash(tuple(self.indices))

    def __call__(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            for i in self.indices:
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = []

        assert isinstance(obj, Sequence), "Sequence expected."

        for i in self.indices:
            if i >= 0:
                while len(obj) <= i:
                    obj.append(None)

            obj[i] = val

        return obj

    def __repr__(self):
        return f"[{','.join(map(str, self.indices))}]"


class GetItems(Step):
    items: Sequence[str]

    def __hash__(self):
        return hash(tuple(self.items))

    def __call__(self, obj: Dict[str, T]) -> Iterable[T]:
        if isinstance(obj, Dict):
            for i in self.items:
                yield obj[i]
        else:
            raise ValueError("Object is not a dictionary.")

    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = dict()

        assert isinstance(obj, Dict), "Dictionary expected."

        for i in self.items:
            obj[i] = val

        return obj

    def __repr__(self):
        return f"[{','.join(self.indices)}]"


class JSONPath(SerialModel):
    """
    Utilitiy class for building JSONPaths.

    Usage:
    
    ```python

        JSONPath().record[5]['somekey]
    ```
    """

    path: Tuple[Step, ...]

    def __init__(self, path: Optional[Tuple[Step, ...]] = None):

        super().__init__(path=path or ())

    def __str__(self):
        return "*" + ("".join(map(repr, self.path)))

    def __repr__(self):
        return "JSONPath()" + ("".join(map(repr, self.path)))

    def __hash__(self):
        return hash(self.path)

    def __len__(self):
        return len(self.path)

    #@staticmethod
    #def parse_obj(d):
    #    path = tuple(map(Step.parse_obj, d['path']))
    #    return JSONPath(path=path)

    def set(self, obj: Any, val: Any) -> Any:
        if len(self.path) == 0:
            return val

        first = self.path[0]
        rest = JSONPath(path=self.path[1:])

        try:
            firsts = first(obj)
            first_obj, firsts = iterable_peek(firsts)

        except (ValueError, IndexError, KeyError, AttributeError):

            # `first` points to an element that does not exist, use `set` to create a spot for it.
            obj = first.set(obj, None)  # will create a spot for `first`
            firsts = first(obj)

        for first_obj in firsts:
            obj = first.set(
                obj,
                rest.set(first_obj, val),
            )

        return obj

    def get_sole_item(self, obj: Any) -> Any:
        return next(self.__call__(obj))

    def __call__(self, obj: Any) -> Iterable[Any]:
        if len(self.path) == 0:
            yield obj
            return

        first = self.path[0]
        if len(self.path) == 1:
            rest = JSONPath(path=())
        else:
            rest = JSONPath(path=self.path[1:])

        for first_selection in first.__call__(obj):
            for rest_selection in rest.__call__(first_selection):
                yield rest_selection

    def _append(self, step: Step) -> JSONPath:
        return JSONPath(path=self.path + (step,))

    def __getitem__(
        self, item: int | str | slice | Sequence[int] | Sequence[str]
    ) -> JSONPath:
        if isinstance(item, int):
            return self._append(GetIndex(index=item))
        if isinstance(item, str):
            return self._append(GetItemOrAttribute(item_or_attribute=item))
        if isinstance(item, slice):
            return self._append(
                GetSlice(start=item.start, stop=item.stop, step=item.step)
            )
        if isinstance(item, Sequence):
            item = tuple(item)
            if all(isinstance(i, int) for i in item):
                return self._append(GetIndices(indices=item))
            elif all(isinstance(i, str) for i in item):
                return self._append(GetItems(items=item))
            else:
                raise TypeError(
                    f"Unhandled sequence item types: {list(map(type, item))}. "
                    f"Note mixing int and str is not allowed."
                )

        raise TypeError(f"Unhandled item type {type(item)}.")

    def __getattr__(self, attr: str) -> JSONPath:
        return self._append(GetItemOrAttribute(item_or_attribute=attr))


# Python utilities


class SingletonPerName():
    """
    Class for creating singleton instances except there being one instance max,
    there is one max per different `name` argument. If `name` is never given,
    reverts to normal singleton behaviour.
    """

    # Hold singleton instances here.
    instances: Dict[Hashable, 'SingletonPerName'] = dict()

    def __new__(cls, name: str = None, *args, **kwargs):
        """
        Create the singleton instance if it doesn't already exist and return it.
        """

        key = cls.__name__, name

        if key not in cls.instances:
            logger.debug(
                f"*** Creating new {cls.__name__} singleton instance for name = {name} ***"
            )
            SingletonPerName.instances[key] = super().__new__(cls)

        return SingletonPerName.instances[key]


# Threading utilities


class TP(SingletonPerName):  # "thread processing"

    # Store here stacks of calls to various thread starting methods so that we can retrieve
    # the trace of calls that caused a thread to start.
    # pre_run_stacks = dict()

    def __init__(self):
        if hasattr(self, "thread_pool"):
            # Already initialized as per SingletonPerName mechanism.
            return

        # TODO(piotrm): if more tasks than `processes` get added, future ones
        # will block and earlier ones may never start executing.
        self.thread_pool = ThreadPool(processes=1024)
        self.running = 0
        self.promises = Queue(maxsize=1024)

    def runrepeatedly(self, func: Callable, rpm: float = 6, *args, **kwargs):

        def runner():
            while True:
                func(*args, **kwargs)
                sleep(60 / rpm)

        self.runlater(runner)

    @staticmethod
    def _thread_target_wrapper(stack, func, *args, **kwargs):
        """
        Wrapper for a function that is started by threads. This is needed to
        record the call stack prior to thread creation as in python threads do
        not inherit the stack. Our instrumentation, however, relies on walking
        the stack and need to do this to the frames prior to thread starts.
        """

        # Keep this for looking up via get_local_in_call_stack .
        pre_start_stack = stack

        return func(*args, **kwargs)

    def _thread_starter(self, func, args, kwargs):
        present_stack = stack()

        prom = self.thread_pool.apply_async(
            self._thread_target_wrapper,
            args=(present_stack, func) + args,
            kwds=kwargs
        )
        return prom

        # thread = Thread(target=func, args=args, kwargs=kwargs)
        # thread.start()
        # self.pre_run_stacks[thread_id] = stack()

    def runlater(self, func: Callable, *args, **kwargs) -> None:
        # prom = self.thread_pool.apply_async(func, args=args, kwds=kwargs)
        prom = self._thread_starter(func, args, kwargs)
        self.promises.put(prom)

    def promise(self, func: Callable[..., T], *args, **kwargs) -> AsyncResult:
        # prom = self.thread_pool.apply_async(func, args=args, kwds=kwargs)
        prom = self._thread_starter(func, args, kwargs)
        self.promises.put(prom)

        return prom

    def finish(self, timeout: Optional[float] = None) -> int:
        logger.debug(f"Finishing {self.promises.qsize()} task(s).")

        timeouts = []

        while not self.promises.empty():
            prom = self.promises.get()
            try:
                prom.get(timeout=timeout)
            except TimeoutError:
                timeouts.append(prom)

        for prom in timeouts:
            self.promises.put(prom)

        if len(timeouts) == 0:
            logger.debug("Done.")
        else:
            logger.debug("Some tasks timed out.")

        return len(timeouts)

    def _status(self) -> List[str]:
        rows = []

        for p in self.thread_pool._pool:
            rows.append([p.is_alive(), str(p)])

        return pd.DataFrame(rows, columns=["alive", "thread"])


def get_local_in_call_stack(
    key: str,
    func: Callable[[Callable], bool],
    offset: int = 1
) -> Optional[Any]:
    """
    Get the value of the local variable named `key` in the stack at the nearest
    frame executing a function which `func` recognizes (returns True on).
    Returns None if `func` does not recognize the correct function. Raises
    RuntimeError if a function is recognized but does not have `key` in its
    locals.

    This method works across threads as long as they are started using the TP
    class above.

    """

    frames = stack()[offset + 1:]  # + 1 to skip this method itself

    # Using queue for frames as additional frames may be added due to handling threads.
    q = Queue()
    for f in frames:
        q.put(f)

    while not q.empty():
        fi = q.get()

        if id(fi.frame.f_code) == id(TP()._thread_target_wrapper.__code__):
            logger.debug(
                "Found thread starter frame. "
                "Will walk over frames in prior to thread start."
            )
            locs = fi.frame.f_locals
            assert "pre_start_stack" in locs, "Pre thread start stack expected but not found."
            for f in locs['pre_start_stack']:
                q.put(f)
            continue

        if func(fi.frame.f_code):
            logger.debug(f"looking {func.__name__} found: {fi}")
            locs = fi.frame.f_locals
            if key in locs:
                return locs[key]
            else:
                raise RuntimeError(f"No local named {key} in {func} found.")
        logger.debug(f"looking {func.__name__}, pass : {fi}")

    return None

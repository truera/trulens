"""
TODO: This file got big. Split off pieces into the utils folder. Already split some:
- utils/text.py -- text outputs/ui related utilities and constants
- utils/python.py -- core python-related utilities

# Utilities.

## Serialization of Python objects

For deferred feedback evaluation, we need to serialize and deserialize python
functions/methods. We feature several storage classes to accomplish this:

Serializable representation | Python thing
----------------------------+------------------
Class                       | (python class)
Module                      | (python module)
Obj                         | (python object)
ObjSerial*                  | (python object)
Function                    | (python function)
Method                      | (python method)

* ObjSerial differs from Obj in that it contains the information necessary to
  reconstruct the object whereas Obj does not. This information is its
  constructor arguments.
"""

from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor as fThreadPoolExecutor
from enum import Enum
import importlib
import inspect
from inspect import stack
import itertools
import json
import logging
from multiprocessing.context import TimeoutError
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import ThreadPool
from pathlib import Path
from pprint import PrettyPrinter
from queue import Queue
from time import sleep
from types import ModuleType
from typing import (
    Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Set,
    Tuple, TypeVar, Union
)

from merkle_json import MerkleJson
from munch import Munch as Bunch
import pandas as pd
import pydantic

from trulens_eval.keys import redact_value

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")

# Optional requirements.

langchain_version = "0.0.230"

REQUIREMENT_LLAMA = (
    "llama_index 0.6.24 or above is required for instrumenting llama_index apps. "
    "Please install it before use: `pip install llama_index>=0.7.0`."
)
REQUIREMENT_LANGCHAIN = (
    f"langchain {langchain_version} or above is required for instrumenting langchain apps. "
    f"Please install it before use: `pip install langchain>={langchain_version}`."
)


class Dummy(object):
    """
    Class to pretend to be a module or some other imported object. Will raise an
    error if accessed in any way.
    """

    def __init__(self, message: str, importer=None):
        self.message = message
        self.importer = importer

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.message)

    def __getattr__(self, name):
        # If in OptionalImport context, create a new dummy for the requested
        # attribute. Otherwise raise error.

        if self.importer is not None and self.importer.importing:
            return Dummy(message=self.message, importer=self.importer)

        raise ModuleNotFoundError(self.message)


class OptionalImports(object):
    """
    Helper context manager for doing multiple imports from an optional module:

    ```python

        with OptionalImports(message='Please install llama_index first'):
            import llama_index
            from llama_index import query_engine

    ```

    The above python block will not raise any errors but once anything else
    about llama_index or query_engine gets accessed, an error is raised with the
    specified message (unless llama_index is installed of course).
    """

    def __init__(self, message: str = None):
        self.message = message
        self.importing = False
        self.imp = builtins.__import__

    def __import__(self, *args, **kwargs):
        try:
            return self.imp(*args, **kwargs)

        except ModuleNotFoundError as e:
            # Check if the import error was from an import in trulens_eval as
            # otherwise we don't want to intercept the error as some modules
            # rely on import failures for various things.
            module_name = inspect.currentframe().f_back.f_globals["__name__"]
            if not module_name.startswith("trulens_eval"):
                raise e
            logger.debug(f"Could not import {args[0]}.")
            return Dummy(message=self.message, importer=self)

    def __enter__(self):
        builtins.__import__ = self.__import__
        self.importing = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.importing = False
        builtins.__import__ = self.imp

        if exc_value is None:
            return None

        print(self.message)
        # Will re-raise exception unless True is returned.
        return None


# Collection utilities


def first(seq: Sequence[T]) -> T:
    return seq[0]


def second(seq: Sequence[T]) -> T:
    return seq[1]


def third(seq: Sequence[T]) -> T:
    return seq[2]


def dict_set_with(dict1: Dict, dict2: Dict):
    dict1.update(dict2)
    return dict1


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

# TODO: rename to "JSON_LIKE" as it is not stringly json.
# JSON = Union[JSON_BASES_T, Sequence['JSON'], Dict[str, 'JSON']]
JSON = Union[JSON_BASES_T, Sequence[Any], Dict[str, Any]]  # Any = JSON

# TODO: rename to "JSON".
JSON_STRICT = Dict[str, JSON]

mj = MerkleJson()


def is_empty(obj):
    try:
        return len(obj) == 0
    except Exception:
        return False


# Key for indicating non-serialized objects in json dumps.
NOSERIO = "__tru_non_serialized_object"


def is_noserio(obj):
    """
    Determines whether the given json object represents some non-serializable
    object. See `noserio`.
    """
    return isinstance(obj, dict) and NOSERIO in obj


def noserio(obj, **extra: Dict) -> dict:
    """
    Create a json structure to represent a non-serializable object. Any
    additional keyword arguments are included.
    """

    inner = Obj.of_object(obj).dict()
    inner.update(extra)

    return {NOSERIO: inner}


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

    # Try the encoders included with pydantic first (should handle things like
    # Datetime):
    try:
        return pydantic.json.pydantic_encoder(obj)
    except:
        # Otherwise give up and indicate a non-serialization.
        return noserio(obj)


# Field/key name used to indicate a circular reference in jsonified objects.
CIRCLE = "__tru_circular_reference"

# Field/key name used to indicate an exception in property retrieval (properties
# execute code in property.fget).
ERROR = "__tru_property_error"

# Key of structure where class information is stored. See WithClassInfo mixin.
CLASS_INFO = "__tru_class_info"

ALL_SPECIAL_KEYS = set([CIRCLE, ERROR, CLASS_INFO, NOSERIO])


def _safe_getattr(obj: Any, k: str) -> Any:
    """
    Try to get the attribute `k` of the given object. This may evaluate some
    code if the attribute is a property and may fail. In that case, an dict
    indicating so is returned.
    """

    v = inspect.getattr_static(obj, k)

    if isinstance(v, property):
        try:
            v = v.fget(obj)
            return v
        except Exception as e:
            return {ERROR: ObjSerial.of_object(e)}
    else:
        return v


def _clean_attributes(obj) -> Dict[str, Any]:
    """
    Determine which attributes of the given object should be enumerated for
    storage and/or display in UI. Returns a dict of those attributes and their
    values.

    For enumerating contents of objects that do not support utility classes like
    pydantic, we use this method to guess what should be enumerated when
    serializing/displaying.
    """

    keys = dir(obj)

    ret = {}

    for k in keys:
        if k.startswith("__"):
            # These are typically very internal components not meant to be
            # exposed beyond immediate definitions. Ignoring these.
            continue

        if k.startswith("_") and k[1:] in keys:
            # Objects often have properties named `name` with their values
            # coming from `_name`. Lets avoid including both the property and
            # the value.
            continue

        v = _safe_getattr(obj, k)
        ret[k] = v

    return ret


# TODO: refactor to somewhere else or change instrument to a generic filter
def jsonify(
    obj: Any,
    dicted: Optional[Dict[int, JSON]] = None,
    instrument: Optional['Instrument'] = None,
    skip_specials: bool = False,
    redact_keys: bool = False
) -> JSON:
    """
    Convert the given object into types that can be serialized in json.

    Args:

        - obj: Any -- the object to jsonify.

        - dicted: Optional[Dict[int, JSON]] -- the mapping from addresses of
          already jsonifed objects (via id) to their json.

        - instrument: Optional[Instrument] -- instrumentation functions for
          checking whether to recur into components of `obj`.

        - skip_specials: bool (default is False) -- if set, will remove
          specially keyed structures from the json. These have keys that start
          with "__tru_".

        - redact_keys: bool (default is False) -- if set, will redact secrets
          from the output. Secrets are detremined by `keys.py:redact_value` .

    Returns:

        JSON | Sequence[JSON]
    """

    from trulens_eval.instruments import Instrument

    instrument = instrument or Instrument()
    dicted = dicted or dict()

    if skip_specials:
        recur_key = lambda k: k not in ALL_SPECIAL_KEYS
    else:
        recur_key = lambda k: True

    if id(obj) in dicted:
        if skip_specials:
            return None
        else:
            return {CIRCLE: id(obj)}

    if isinstance(obj, JSON_BASES):
        if redact_keys and isinstance(obj, str):
            return redact_value(obj)
        else:
            return obj

    if isinstance(obj, Path):
        return str(obj)

    if type(obj) in pydantic.json.ENCODERS_BY_TYPE:
        return obj

    # TODO: should we include duplicates? If so, dicted needs to be adjusted.
    new_dicted = {k: v for k, v in dicted.items()}

    recur = lambda o: jsonify(
        obj=o,
        dicted=new_dicted,
        instrument=instrument,
        skip_specials=skip_specials,
        redact_keys=redact_keys
    )

    if isinstance(obj, Enum):
        return obj.name

    if isinstance(obj, Dict):
        temp = {}
        new_dicted[id(obj)] = temp
        temp.update({k: recur(v) for k, v in obj.items() if recur_key(k)})

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in temp.items():
                temp[k] = redact_value(v=v, k=k)

        return temp

    elif isinstance(obj, Sequence):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (recur(v) for v in obj):
            temp.append(x)
        return temp

    elif isinstance(obj, Set):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (recur(v) for v in obj):
            temp.append(x)
        return temp

    elif isinstance(obj, pydantic.BaseModel):
        # Not even trying to use pydantic.dict here.

        temp = {}
        new_dicted[id(obj)] = temp
        temp.update(
            {
                k: recur(_safe_getattr(obj, k))
                for k, v in obj.__fields__.items()
                if not v.field_info.exclude and recur_key(k)
            }
        )

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in temp.items():
                temp[k] = redact_value(v=v, k=k)

        if instrument.to_instrument_object(obj):
            temp[CLASS_INFO] = Class.of_class(
                cls=obj.__class__, with_bases=True
            ).dict()

        return temp

    elif obj.__class__.__module__.startswith("llama_index."):
        # Most of llama_index classes do not inherit a storage-utility class
        # like pydantc so we have to enumerate their contents ourselves based on
        # some heuristics.

        temp = {}
        new_dicted[id(obj)] = temp

        kvs = _clean_attributes(obj)

        temp.update(
            {
                k: recur(v) for k, v in kvs.items() if recur_key(k) and (
                    isinstance(v, JSON_BASES) or isinstance(v, Dict) or
                    isinstance(v, Sequence) or
                    instrument.to_instrument_object(v)
                )
            }
        )

        if instrument.to_instrument_object(obj):
            temp[CLASS_INFO] = Class.of_class(
                cls=obj.__class__, with_bases=True
            ).dict()

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

    yield (query, obj)

    if isinstance(obj, JSON_BASES):
        pass

    elif isinstance(obj, pydantic.BaseModel):
        for k in obj.__fields__:
            v = getattr(obj, k)
            sub_query = query[k]
            for res in all_objects(v, sub_query):
                yield res

    elif isinstance(obj, Dict):
        for k, v in obj.items():
            sub_query = query[k]
            for res in all_objects(obj[k], sub_query):
                yield res

    elif isinstance(obj, Sequence):
        for i, v in enumerate(obj):
            sub_query = query[i]
            for res in all_objects(obj[i], sub_query):
                yield res

    elif isinstance(obj, Iterable):
        pass
        # print(f"Cannot create query for Iterable types like {obj.__class__.__name__} at query {query}. Convert the iterable to a sequence first.")

    else:
        pass
        # print(f"Unhandled object type {obj} {type(obj)}")


def leafs(obj: Any) -> Iterable[Tuple[str, Any]]:
    for q in leaf_queries(obj):
        path_str = str(q)
        val = q(obj)
        yield (path_str, val)


def matching_objects(obj: Any,
                     match: Callable) -> Iterable[Tuple[JSONPath, Any]]:
    for q, val in all_objects(obj):
        if match(q, val):
            yield (q, val)


def matching_queries(obj: Any, match: Callable) -> Iterable[JSONPath]:
    for q, _ in matching_objects(obj, match=match):
        yield q


class SerialModel(pydantic.BaseModel):
    """
    Trulens-specific additions on top of pydantic models. Includes utilities to
    help serialization mostly.
    """

    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        if isinstance(obj, dict):
            if CLASS_INFO in obj:

                cls = Class(**obj[CLASS_INFO])
                del obj[CLASS_INFO]
                model = cls.model_validate(obj, **kwargs)

                return WithClassInfo.of_model(model=model, cls=cls)
            else:
                print(
                    f"Warning: May not be able to properly reconstruct object {obj}."
                )

                return super().model_validate(obj, **kwargs)

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
        return hash(self.item_or_attribute)

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

    def is_immediate_prefix_of(self, other: JSONPath):
        return self.is_prefix_of(other) and len(self.path) + 1 == len(
            other.path
        )

    def is_prefix_of(self, other: JSONPath):
        p = self.path
        pother = other.path

        if len(p) > len(pother):
            return False

        for s1, s2 in zip(p, pother):
            if s1 != s2:
                return False

        return True

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

        k = cls.__name__, name

        if k not in cls.instances:
            logger.debug(
                f"*** Creating new {cls.__name__} singleton instance for name = {name} ***"
            )
            SingletonPerName.instances[k] = super().__new__(cls)

        return SingletonPerName.instances[k]


# Threading utilities


def _future_target_wrapper(stack, func, *args, **kwargs):
    """
    Wrapper for a function that is started by threads. This is needed to
    record the call stack prior to thread creation as in python threads do
    not inherit the stack. Our instrumentation, however, relies on walking
    the stack and need to do this to the frames prior to thread starts.
    """

    # Keep this for looking up via get_local_in_call_stack .
    pre_start_stack = stack

    return func(*args, **kwargs)


class ThreadPoolExecutor(fThreadPoolExecutor):

    def submit(self, fn, /, *args, **kwargs):
        present_stack = stack()
        return super().submit(
            _future_target_wrapper, present_stack, fn, *args, **kwargs
        )


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

    def _thread_starter(self, func, args, kwargs):
        present_stack = stack()

        prom = self.thread_pool.apply_async(
            _future_target_wrapper,
            args=(present_stack, func) + args,
            kwds=kwargs
        )
        return prom

    def runlater(self, func: Callable, *args, **kwargs) -> None:
        prom = self._thread_starter(func, args, kwargs)
        self.promises.put(prom)

    def promise(self, func: Callable[..., T], *args, **kwargs) -> AsyncResult:
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


# python instrumentation utilities


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

        logger.debug(f"{fi.frame.f_code}")

        if id(fi.frame.f_code) == id(_future_target_wrapper.__code__):
            logger.debug(
                "Found thread starter frame. "
                "Will walk over frames prior to thread start."
            )
            locs = fi.frame.f_locals
            assert "pre_start_stack" in locs, "Pre thread start stack expected but not found."
            for f in locs['pre_start_stack']:
                q.put(f)
            continue

        if func(fi.frame.f_code):
            logger.debug(f"looking via {func.__name__}; found {fi}")
            locs = fi.frame.f_locals
            if key in locs:
                return locs[key]
            else:
                raise RuntimeError(f"No local named {key} found.")

    return None


class Module(SerialModel):
    package_name: Optional[str]  # some modules are not in a package
    module_name: str

    def of_module(mod: ModuleType, loadable: bool = False) -> 'Module':
        if loadable and mod.__name__ == "__main__":
            # running in notebook
            raise ImportError(f"Module {mod} is not importable.")

        return Module(package_name=mod.__package__, module_name=mod.__name__)

    def of_module_name(module_name: str, loadable: bool = False) -> 'Module':
        if loadable and module_name == "__main__":
            # running in notebook
            raise ImportError(f"Module {module_name} is not importable.")

        mod = importlib.import_module(module_name)
        package_name = mod.__package__
        return Module(package_name=package_name, module_name=module_name)

    def load(self) -> ModuleType:
        return importlib.import_module(
            self.module_name, package=self.package_name
        )


class Class(SerialModel):
    """
    A python class. Should be enough to deserialize the constructor. Also
    includes bases so that we can query subtyping relationships without
    deserializing the class first.
    """

    name: str

    module: Module

    bases: Optional[Sequence[Class]]

    def __repr__(self):
        return self.module.module_name + "." + self.name

    def __str__(self):
        return f"{self.name}({self.module.module_name})"

    def base_class(self) -> 'Class':
        """
        Get the deepest base class in the same module as this class.
        """
        module_name = self.module.module_name
        for base in self.bases[::-1]:
            if base.module.module_name == module_name:
                return base

        return self

    def _check_importable(self):
        try:
            cls = self.load()
        except Exception as e:
            raise ImportError(
                f"Class {self} is not importable. "
                "If you are defining custom feedback function implementations, make sure they can be imported by python scripts. "
                "If you defined a function in a notebook, it will not be importable."
            )

    @staticmethod
    def of_class(
        cls: type, with_bases: bool = False, loadable: bool = False
    ) -> 'Class':
        ret = Class(
            name=cls.__name__,
            module=Module.of_module_name(cls.__module__, loadable=loadable),
            bases=list(map(lambda base: Class.of_class(cls=base), cls.__mro__))
            if with_bases else None
        )
        if loadable:
            ret._check_importable()

        return ret

    @staticmethod
    def of_object(
        obj: object, with_bases: bool = False, loadable: bool = False
    ):
        return Class.of_class(
            cls=obj.__class__, with_bases=with_bases, loadable=loadable
        )

    @staticmethod
    def of_json(json: JSON):
        assert CLASS_INFO in json, "Class info not in json."

        return Class(**json[CLASS_INFO])

    def load(self) -> type:  # class
        try:
            mod = self.module.load()
            return getattr(mod, self.name)

        except Exception as e:
            raise RuntimeError(f"Could not load class {self} because {e}.")

    def noserio_issubclass(self, class_name: str, module_name: str):
        bases = self.bases

        assert bases is not None, "Cannot do subclass check without bases. Serialize me with `Class.of_class(with_bases=True ...)`."

        for base in bases:
            if base.name == class_name and base.module.module_name == module_name:
                return True

        return False


# inspect.signature does not work on builtin type constructors but they are used
# like this method. Use it to create a signature of a builtin constructor.
def builtin_init_dummy(self, *args, **kwargs):
    pass


builtin_init_sig = inspect.signature(builtin_init_dummy)


class Obj(SerialModel):
    # TODO: refactor this into something like WithClassInfo, perhaps
    # WithObjectInfo, and store required constructor inputs as attributes with
    # potentially a placeholder for additional arguments that are not
    # attributes, under a special key like "__tru_object_info".
    """
    An object that may or may not be serializable. Do not use for base types
    that don't have a class.
    """

    cls: Class

    # From id(obj), identifies memory location of a python object. Use this for
    # handling loops in JSON objects.
    id: int

    @classmethod
    def validate(cls, d) -> 'Obj':
        if isinstance(d, Obj):
            return d
        elif isinstance(d, ObjSerial):
            return d
        elif isinstance(d, Dict):
            return Obj.pick(**d)
        else:
            raise RuntimeError(f"Unhandled Obj source of type {type(d)}.")

    @staticmethod
    def pick(**d):
        if 'init_bindings' in d:
            return ObjSerial(**d)
        else:
            return Obj(**d)

    @staticmethod
    def of_object(
        obj: object,
        cls: Optional[type] = None,
        loadable: bool = False
    ) -> Union['Obj', 'ObjSerial']:
        if loadable:
            return ObjSerial.of_object(obj=obj, cls=cls, loadable=loadable)

        if cls is None:
            cls = obj.__class__

        return Obj(cls=Class.of_class(cls), id=id(obj))

    def load(self) -> object:
        # Check that the object's class is importable before the other error is thrown.
        self.cls._check_importable()

        raise RuntimeError(
            f"Trying to load an object without constructor arguments: {pp.pformat(self.dict())}."
        )


class Bindings(SerialModel):
    args: Tuple
    kwargs: Dict[str, Any]

    @staticmethod
    def of_bound_arguments(b: inspect.BoundArguments) -> Bindings:
        return Bindings(args=b.args, kwargs=b.kwargs)

    def load(self, sig: inspect.Signature):
        return sig.bind(*self.args, **self.kwargs)


def _safe_init_sig(cls):
    """
    Get the signature of the constructor method of the given class `cls`. If it is
    a builtin class, this typically raises an exeception in which case we return
    a generic signature that seems typical of builtin constructors.
    """

    try:
        return inspect.signature(cls)
    except Exception as e:
        return builtin_init_sig


class ObjSerial(Obj):
    """
    Object that can be deserialized, or at least intended to be deserialized.
    Stores additional information beyond the class that can be used to
    deserialize it, the constructor bindings.
    """

    init_bindings: Bindings

    @staticmethod
    def of_object(obj: object, cls: Optional[type] = None) -> 'Obj':
        if cls is None:
            cls = obj.__class__

        # Constructor arguments for some common types.
        if isinstance(obj, pydantic.BaseModel):
            init_args = ()
            init_kwargs = obj.dict()
        elif isinstance(obj, Exception):
            init_args = obj.args
            init_kwargs = {}
        else:
            # For unknown types, check if the constructor for cls expect
            # arguments and fail if so as we don't know where to get them. If
            # there are none, create empty init bindings.

            sig = _safe_init_sig(cls)
            if len(sig.parameters) > 0:
                raise RuntimeError(
                    f"Do not know how to get constructor arguments for object of type {cls.__name__}. "
                    f"If you are defining a custom feedback function, define its implementation as a function or a method of a Provider subclass."
                )

            init_args = ()
            init_kwargs = {}

        # TODO: dataclasses
        # TODO: dataclasses_json

        sig = _safe_init_sig(cls.__call__)
        # NOTE: Something related to pydantic models incorrectly sets signature
        # of cls so we need to check cls.__call__ instead.

        b = sig.bind(*init_args, **init_kwargs)
        bindings = Bindings.of_bound_arguments(b)

        cls_serial = Class.of_class(cls)
        cls_serial._check_importable()

        return ObjSerial(cls=cls_serial, id=id(obj), init_bindings=bindings)

    def load(self) -> object:
        cls = self.cls.load()

        sig = _safe_init_sig(cls)
        bindings = self.init_bindings.load(sig)

        return cls(*bindings.args, **bindings.kwargs)


class FunctionOrMethod(SerialModel):

    @staticmethod
    def pick(**kwargs):
        # Temporary hack to deserialization of a class with more than one subclass.

        if 'obj' in kwargs:
            return Method(**kwargs)

        elif 'cls' in kwargs:
            return Function(**kwargs)

    @classmethod
    def __get_validator__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, d) -> 'FunctionOrMethod':

        if isinstance(d, Dict):
            return FunctionOrMethod.pick(**d)
        else:
            return d

    @staticmethod
    def of_callable(c: Callable, loadable: bool = False) -> 'FunctionOrMethod':
        """
        Serialize the given callable. If `loadable` is set, tries to add enough
        info for the callable to be deserialized.
        """

        if inspect.ismethod(c):
            self = c.__self__
            return Method.of_method(c, obj=self, loadable=loadable)

        else:

            return Function.of_function(c, loadable=loadable)

    def load(self) -> Callable:
        raise NotImplementedError()


class Method(FunctionOrMethod):
    """
    A python method. A method belongs to some class in some module and must have
    a pre-bound self object. The location of the method is encoded in `obj`
    alongside self. If obj is ObjSerial, this method should be deserializable.
    """

    obj: Obj
    name: str

    @staticmethod
    def of_method(
        meth: Callable,
        cls: Optional[type] = None,
        obj: Optional[object] = None,
        loadable: bool = False
    ) -> 'Method':
        if obj is None:
            assert inspect.ismethod(
                meth
            ), f"Expected a method (maybe it is a function?): {meth}"
            obj = meth.__self__

        if cls is None:
            if isinstance(cls, type):
                # classmethod, self is a type
                cls = obj
            else:
                # normal method, self is instance of cls
                cls = obj.__class__

        obj_json = (ObjSerial if loadable else Obj).of_object(obj, cls=cls)

        return Method(obj=obj_json, name=meth.__name__)

    def load(self) -> Callable:
        obj = self.obj.load()
        return getattr(obj, self.name)


class Function(FunctionOrMethod):
    """
    A python function. Could be a static method inside a class (not instance of
    the class).
    """

    module: Module

    # For static methods in a class which we view as functions, not yet
    # supported:
    cls: Optional[Class]

    name: str

    @staticmethod
    def of_function(
        func: Callable,
        module: Optional[ModuleType] = None,
        cls: Optional[type] = None,
        loadable: bool = False
    ) -> 'Function':  # actually: class

        if module is None:
            module = Module.of_module_name(func.__module__, loadable=loadable)

        if cls is not None:
            cls = Class.of_class(cls, loadable=loadable)

        return Function(cls=cls, module=module, name=func.__name__)

    def load(self) -> Callable:
        if self.cls is not None:
            # TODO: static/class methods work in progress

            cls = self.cls.load()  # does not create object instance
            return getattr(cls, self.name)  # lookup static/class method

        else:
            mod = self.module.load()
            try:
                return getattr(mod, self.name)  # function not inside a class
            except Exception:
                raise ImportError(
                    f"Function {self} is not importable. "
                    "If you are defining custom feedback function implementations, make sure they can be imported by python scripts. "
                    "If you defined a function in a notebook, it will not be importable."
                )


class WithClassInfo(pydantic.BaseModel):
    """
    Mixin to track class information to aid in querying serialized components
    without having to load them.
    """

    # Using this odd key to not pollute attribute names in whatever class we mix
    # this into. Should be the same as CLASS_INFO.
    __tru_class_info: Class

    def __init__(
        self,
        class_info: Optional[Class] = None,
        obj: Optional[object] = None,
        cls: Optional[type] = None,
        **kwargs
    ):
        if obj is not None:
            cls = type(obj)

        if class_info is None:
            assert cls is not None, "Either `class_info`, `obj` or `cls` need to be specified."
            class_info = Class.of_class(cls, with_bases=True)

        super().__init__(__tru_class_info=class_info, **kwargs)

    @staticmethod
    def of_object(obj: object):
        return WithClassInfo(class_info=Class.of_class(obj.__class__))

    @staticmethod
    def of_class(cls: type):  # class
        return WithClassInfo(class_info=Class.of_class(cls))

    @staticmethod
    def of_model(model: pydantic.BaseModel, cls: Class):
        return WithClassInfo(class_info=cls, **model.dict())


def get_owner_of_method(cls, method_name) -> type:
    """
    Get the actual defining class of the given method whether it is cls or one
    of its parent classes.
    """

    # TODO

    return cls

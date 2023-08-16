"""
Serialization utilities mostly with regards to JSON.
"""

from __future__ import annotations

from enum import Enum
import inspect
from inspect import signature
import json
import logging
from pathlib import Path
from pprint import PrettyPrinter
from typing import (Any, Callable, Dict, Iterable, Optional, Sequence, Set,
                    Tuple, Union)

from merkle_json import MerkleJson
from munch import Munch as Bunch
import pydantic

logger = logging.getLogger(__name__)
pp = PrettyPrinter()


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


class SerialModel(pydantic.BaseModel):
    """
    Trulens-specific additions on top of pydantic models. Includes utilities to
    help serialization mostly.
    """

    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        # import hierarchy circle here

        from trulens_eval.keys import redact_value
        from trulens_eval.utils.pyschema import Class
        from trulens_eval.utils.pyschema import WithClassInfo

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
            setattr(obj, self.item_or_attribute, val)

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
        return f"[{','.join(self.items)}]"


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

    def __add__(self, other: JSONPath):
        return JSONPath(path=self.path + other.path)

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

def callable_name(c: Callable):
    if hasattr(c, "__name__"):
        return c.__name__
    elif hasattr(c, "__call__"):
        return callable_name(c.__call__)
    else:
        return str(c)

def safe_signature(func_or_obj: Any):
    if hasattr(func_or_obj, "__call__"):
        # If given an obj that is callable (has __call__ defined), we want to
        # return signature of that call instead of letting inspect.signature
        # explore that object further. Doing so may produce exceptions due to
        # contents of those objects producing exceptions when attempting to
        # retrieve them.

        return signature(func_or_obj.__call__)

    else:
        assert isinstance(func_or_obj, Callable), f"Expected a Callable. Got {type(func_or_obj)} instead."

        return signature(func_or_obj)


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

    content = None

    if isinstance(obj, Enum):
        content = obj.name

    elif isinstance(obj, Dict):
        temp = {}
        new_dicted[id(obj)] = temp
        temp.update({k: recur(v) for k, v in obj.items() if recur_key(k)})

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in temp.items():
                temp[k] = redact_value(v=v, k=k)

        content = temp

    elif isinstance(obj, Sequence):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (recur(v) for v in obj):
            temp.append(x)

        content = temp

    elif isinstance(obj, Set):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (recur(v) for v in obj):
            temp.append(x)

        content = temp

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

        content = temp

    elif instrument.to_instrument_object(obj):

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

        content = temp

    else:
        logger.debug(
            f"Do not know how to jsonify an object '{str(obj)[0:32]}' of type '{type(obj)}'."
        )

        content = noserio(obj)

    # Add class information for objects that are to be instrumented, known as
    # "components".
    if instrument.to_instrument_object(obj):
        content[CLASS_INFO] = Class.of_class(
            cls=obj.__class__, with_bases=True
        ).dict()

    if not isinstance(obj, JSONPath) and hasattr(obj, "jsonify_extra"):
        # Problem with JSONPath and similar objects: they always say they have every attribute.

        content = obj.jsonify_extra(content)

    return content


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

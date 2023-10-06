"""
Serialization utilities.
"""

from __future__ import annotations

import ast
from ast import dump
from ast import parse
import logging
from pprint import PrettyPrinter
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Set, Tuple, TypeVar, Union)

from merkle_json import MerkleJson
from munch import Munch as Bunch
import pydantic

from trulens_eval.utils.containers import iterable_peek

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")

# JSON types

JSON_BASES = (str, int, float, type(None))
JSON_BASES_T = Union[str, int, float, type(None)]

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
        from trulens_eval.utils.pyschema import Class
        from trulens_eval.utils.pyschema import CLASS_INFO
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


class SerialBytes(pydantic.BaseModel):
    # Raw data that we want to nonetheless serialize .
    data: bytes

    def dict(self):
        import base64
        encoded = base64.b64encode(self.data)
        return dict(data=encoded)

    @classmethod
    def parse_obj(cls, d: Any):
        import base64

        if isinstance(d, Dict):
            encoded = d['data']
            return SerialBytes(data=base64.b64decode(encoded))
        else:
            raise ValueError(d)


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

    # NOTE: We also allow to lookup elements within sequences if the subelements
    # have the item or attribute. We issue warning if this is ambiguous (looking
    # up in a sequence of more than 1 element).

    item_or_attribute: str  # distinct from "item" for deserialization

    def __hash__(self):
        return hash(self.item_or_attribute)

    def __call__(self, obj: Dict[str, T]) -> Iterable[T]:
        # Special handling of sequences. See NOTE above.
        if isinstance(obj, Sequence):
            if len(obj) == 1:
                for r in self.__call__(obj=obj[0]):
                    yield r
            elif len(obj) == 0:
                raise ValueError(
                    f"Object not a dictionary or sequence of dictionaries: {obj}."
                )
            else:  # len(obj) > 1
                logger.warning(
                    f"Object (of type {type(obj).__name__}) is a sequence containing more than one dictionary. "
                    f"Lookup by item or attribute `{self.item_or_attribute}` is ambiguous. "
                    f"Use a lookup by index(es) or slice first to disambiguate."
                )
                for r in self.__call__(obj=obj[0]):
                    yield r

        # Otherwise handle a dict or object with the named attribute.
        elif isinstance(obj, Dict):
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


class ParseException(Exception):

    def __init__(self, exp_string: str, exp_ast: ast.AST):
        self.exp_string = exp_string
        self.exp_ast = exp_ast

    def __str__(self):
        return f"Failed to parse expression `{self.exp_string}` as a `JSONPath`.\n{dump(self.exp_ast) if self.exp_ast is not None else 'AST is None'}"


class JSONPath(SerialModel):
    """
    Utilitiy class for building JSONPaths.

    **Usage:**
    
    ```python

        JSONPath().record[5]['somekey]
    ```
    """

    path: Tuple[Step, ...]

    def __init__(self, path: Optional[Tuple[Step, ...]] = None):

        super().__init__(path=path or ())

    @staticmethod
    def of_string(s: str) -> 'JSONPath':
        if len(s) == 0:
            return JSONPath()

        try:
            exp = parse(f"PLACEHOLDER.{s}", mode="eval")
        except SyntaxError as e:
            raise ParseException(s, None)

        if not isinstance(exp, ast.Expression):
            raise ParseException(s, exp)

        exp = exp.body

        path = []

        def of_index(idx):
            if isinstance(idx, ast.Tuple):
                elts = tuple(of_index(elt.value) for elt in idx.elts)
                if all(isinstance(e, GetItem) for e in elts):
                    return GetItems(items=tuple(e.item for e in elts))
                elif all(isinstance(e, int) for e in elts):
                    return GetIndices(indices=tuple(e.index for e in elts))
                else:
                    raise ParseException(s, idx)

            elif isinstance(idx, ast.Constant):
                if isinstance(idx.value, str):
                    return GetItem(item=idx.value)
                elif isinstance(idx.value, int):
                    return GetIndex(index=idx.value)
                else:
                    raise ParseException(s, idx)

            elif isinstance(idx, ast.UnaryOp):
                if isinstance(idx.op, ast.USub):
                    oper = of_index(idx.operand)
                    if not isinstance(oper, GetIndex):
                        raise ParseException(s, idx)
                    return GetIndex(index=-oper.index)

            elif idx is None:
                return None

            else:
                raise ParseException(s, exp)

        while exp is not None:
            if isinstance(exp, ast.Attribute):
                attr_name = exp.attr
                path.append(GetItemOrAttribute(item_or_attribute=attr_name))
                exp = exp.value

            elif isinstance(exp, ast.Subscript):
                sub = exp.slice

                if isinstance(sub, ast.Index):
                    idx = sub.value
                    step = of_index(idx)
                    path.append(step)

                elif isinstance(sub, ast.Slice):
                    vals = tuple(
                        of_index(v) for v in (sub.lower, sub.upper, sub.step)
                    )

                    if not all(
                            e is None or isinstance(e, GetIndex) for e in vals):
                        raise ParseException(s, exp)

                    vals = tuple(None if e is None else e.index for e in vals)
                    path.append(
                        GetSlice(start=vals[0], stop=vals[1], step=vals[2])
                    )

                else:
                    raise ParseException(s, exp)

                exp = exp.value
            elif isinstance(exp, ast.Name):
                if exp.id != "PLACEHOLDER":
                    raise ParseException(s, exp)

                exp = None
            else:
                raise ParseException(s, exp)

        return JSONPath(path=path[::-1])

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

    def set_or_append(self, obj: Any, val: Any) -> Any:
        """
        If `obj` at path `self` is None or does not exist, sets it to a list
        containing only the given `val`. If it already exists as a sequence,
        appends `val` to that sequence as a list. If it is set but not a sequence,
        error is thrown.
        
        """
        try:
            existing = self.get_sole_item(obj)
            if isinstance(existing, Sequence):
                return self.set(obj, list(existing) + [val])
            elif existing is None:
                return self.set(obj, [val])
            else:
                raise ValueError(
                    f"Trying to append to object which is not a list; "
                    f"is of type {type(existing).__name__} instead."
                )

        except Exception:
            return self.set(obj, [val])

    def set(self, obj: Any, val: Any) -> Any:
        """
        In `obj` at path `self` exists, change it to `val`. Otherwise create a
        spot for it with Munch objects and then set it.
        """

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

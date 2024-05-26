"""
Serialization utilities.

TODO: Lens class: can we store just the python AST instead of building up our
own "Step" classes to hold the same data? We are already using AST for parsing.
"""

from __future__ import annotations

import ast
from ast import dump
from ast import parse
from contextvars import ContextVar
from copy import copy
import logging
from typing import (
    Any, Callable, ClassVar, Dict, Generic, Hashable, Iterable, List, Optional,
    Sequence, Set, Sized, Tuple, TypeVar, Union
)

from merkle_json import MerkleJson
from munch import Munch as Bunch
import pydantic
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from pydantic_core import CoreSchema
import rich

from trulens_eval.utils.containers import iterable_peek
from trulens_eval.utils.python import class_name

logger = logging.getLogger(__name__)

T = TypeVar("T")

JSON_BASES: Tuple[type, ...] = (str, int, float, bytes, type(None))
"""
Tuple of JSON-able base types.

Can be used in `isinstance` checks.
"""

JSON_BASES_T = Union[\
    str, int, float, bytes, None
                    ]
"""
Alias for JSON-able base types.
"""

JSON = Union[\
    JSON_BASES_T,
    Sequence[Any],
    Dict[str, Any]
            ]
"""Alias for (non-strict) JSON-able data (`Any` = `JSON`).

If used with type argument, that argument indicates what the JSON represents and
can be desererialized into.

Formal JSON must be a `dict` at the root but non-strict here means that the root
can be a basic type or a sequence as well.
"""

JSON_STRICT = Dict[str, JSON]
"""
Alias for (strictly) JSON-able data.

Python object that is directly mappable to JSON.
"""


class JSONized(dict, Generic[T]):  # really JSON_STRICT
    """JSON-encoded data the can be deserialized into a given type `T`.
    
    This class is meant only for type annotations. Any
    serialization/deserialization logic is handled by different classes, usually
    subclasses of `pydantic.BaseModel`.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Make pydantic treat this class same as a `dict`."""
        return handler(core_schema.dict_schema())


mj = MerkleJson()


def model_dump(obj: Union[pydantic.BaseModel, pydantic.v1.BaseModel]) -> dict:
    """
    Return the dict/model_dump of the given pydantic instance regardless of it
    being v2 or v1.
    """

    if isinstance(obj, pydantic.BaseModel):
        return obj.model_dump()
    elif isinstance(obj, pydantic.v1.BaseModel):
        return obj.dict()
    else:
        raise ValueError("Not a pydantic.BaseModel.")


class SerialModel(pydantic.BaseModel):
    """
    Trulens-specific additions on top of pydantic models. Includes utilities to
    help serialization mostly.
    """

    formatted_objects: ClassVar[ContextVar[Set[int]]
                               ] = ContextVar("formatted_objects")

    def __rich_repr__(self) -> rich.repr.Result:
        """Requirement for pretty printing using the rich package."""

        # yield class_name(type(self))

        # If this is a root repr, create a new set for already-formatted objects.
        tok = None
        if SerialModel.formatted_objects.get(None) is None:
            tok = SerialModel.formatted_objects.set(set())

        formatted_objects = SerialModel.formatted_objects.get()

        if formatted_objects is None:
            formatted_objects = set()

        if id(self) in formatted_objects:
            yield f"{class_name(type(self))}@0x{id(self):x}"

            if tok is not None:
                SerialModel.formatted_objects.reset(tok)

            return

        formatted_objects.add(id(self))

        for k, v in self.__dict__.items():
            # This might result in recursive calls to __rich_repr__ of v.
            yield k, v

        if tok is not None:
            SerialModel.formatted_objects.reset(tok)

    def model_dump_json(self, **kwargs):
        from trulens_eval.utils.json import json_str_of_obj

        return json_str_of_obj(self, **kwargs)

    def model_dump(self, **kwargs):
        from trulens_eval.utils.json import jsonify

        return jsonify(self, **kwargs)

    # NOTE(piotrm): regaring model_validate: custom deserialization is done in
    # WithClassInfo class but only for classes that mix it in.

    def update(self, **d):
        for k, v in d.items():
            setattr(self, k, v)

        return self

    def replace(self, **d):
        copy = self.model_copy()
        copy.update(**d)
        return copy


class SerialBytes(pydantic.BaseModel):
    # Raw data that we want to nonetheless serialize .
    data: bytes

    def dict(self):
        import base64
        encoded = base64.b64encode(self.data)
        return dict(data=encoded)

    def __init__(self, data: Union[str, bytes]):
        super().__init__(data=data)

    @classmethod
    def model_validate(cls, obj, **kwargs):
        import base64

        if isinstance(obj, Dict):
            encoded = obj['data']
            if isinstance(encoded, str):
                return SerialBytes(data=base64.b64decode(encoded))
            elif isinstance(encoded, bytes):
                return SerialBytes(data=encoded)
            else:
                raise ValueError(obj)
        else:
            raise ValueError(obj)


# NOTE1: Lens, a container for selector/accessors/setters of data stored in a
# json structure. Cannot make abstract since pydantic will try to initialize it.
class Step(pydantic.BaseModel, Hashable):
    """
    A step in a selection path.
    """

    def __hash__(self):
        raise TypeError(f"Should never be called, self={self.model_dump()}")

    @classmethod
    def model_validate(cls, obj, **kwargs):

        if isinstance(obj, Step):
            return super().model_validate(obj, **kwargs)

        elif isinstance(obj, dict):

            ATTRIBUTE_TYPE_MAP = {
                'item': GetItem,
                'index': GetIndex,
                'attribute': GetAttribute,
                'item_or_attribute': GetItemOrAttribute,
                'start': GetSlice,
                'stop': GetSlice,
                'step': GetSlice,
                'items': GetItems,
                'indices': GetIndices,
                'collect': Collect
            }

            a = next(iter(obj.keys()))
            if a in ATTRIBUTE_TYPE_MAP:
                return ATTRIBUTE_TYPE_MAP[a](**obj)

        raise RuntimeError(
            f"Do not know how to interpret {obj} as a `Lens` `Step`."
        )

    # @abc.abstractmethod # NOTE1
    def get(self, obj: Any) -> Iterable[Any]:
        """
        Get the element of `obj`, indexed by `self`.
        """
        raise NotImplementedError()

    # @abc.abstractmethod # NOTE1
    def set(self, obj: Any, val: Any) -> Any:
        """
        Set the value(s) indicated by self in `obj` to value `val`.
        """
        raise NotImplementedError()


class Collect(Step):
    # Need something for `Step.validate` to tell that it is looking at Collect.
    collect: None = None

    # Hashable requirement.
    def __hash__(self):
        return hash("collect")

    # Step requirement.
    def get(self, obj: Any) -> Iterable[List[Any]]:
        # Needs to be handled in Lens class itself.
        raise NotImplementedError()

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        raise NotImplementedError()

    def __repr__(self):
        return f".collect()"


class StepItemOrAttribute(Step):
    # NOTE1
    def get_item_or_attribute(self):
        raise NotImplementedError()


class GetAttribute(StepItemOrAttribute):
    attribute: str

    # Hashable requirement.
    def __hash__(self):
        return hash(self.attribute)

    # StepItemOrAttribute requirement
    def get_item_or_attribute(self):
        return self.attribute

    # Step requirement
    def get(self, obj: Any) -> Iterable[Any]:
        if hasattr(obj, self.attribute):
            yield getattr(obj, self.attribute)
        else:
            raise ValueError(
                f"Object {obj} does not have attribute: {self.attribute}"
            )

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = Bunch()

        # might cause isses
        obj = copy(obj)

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

    # Hashable requirement
    def __hash__(self):
        return hash(self.index)

    # Step requirement
    def get(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            if len(obj) > self.index:
                yield obj[self.index]
            else:
                raise IndexError(f"Index out of bounds: {self.index}")
        else:
            raise ValueError(f"Object {obj} is not a sequence.")

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = []

        assert isinstance(obj, Sequence), "Sequence expected."

        # copy
        obj = list(obj)

        if self.index >= 0:
            while len(obj) <= self.index:
                obj.append(None)

        obj[self.index] = val
        return obj

    def __repr__(self):
        return f"[{self.index}]"


class GetItem(StepItemOrAttribute):
    item: str

    # Hashable requirement
    def __hash__(self):
        return hash(self.item)

    # StepItemOrAttribute requirement
    def get_item_or_attribute(self):
        return self.item

    # Step requirement
    def get(self, obj: Dict[str, T]) -> Iterable[T]:
        if isinstance(obj, Dict):
            if self.item in obj:
                yield obj[self.item]
            else:
                raise KeyError(f"Key not in dictionary: {self.item}")
        else:
            raise ValueError(f"Object {obj} is not a dictionary.")

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = dict()

        assert isinstance(obj, Dict), "Dictionary expected."

        # copy
        obj = {k: v for k, v in obj.items()}

        obj[self.item] = val
        return obj

    def __repr__(self):
        return f"[{repr(self.item)}]"


class GetItemOrAttribute(StepItemOrAttribute):
    """A step in a path lens that selects an item or an attribute.

    !!! note:
        _TruLens-Eval_ allows lookuping elements within sequences if the subelements
        have the item or attribute. We issue warning if this is ambiguous (looking
        up in a sequence of more than 1 element).
    """

    item_or_attribute: str  # distinct from "item" for deserialization

    # Hashable requirement
    def __hash__(self):
        return hash(self.item_or_attribute)

    # StepItemOrAttribute requirement
    def get_item_or_attribute(self):
        return self.item_or_attribute

    # Step requirement
    def get(self, obj: Dict[str, T]) -> Iterable[T]:
        # Special handling of sequences. See NOTE above.

        if isinstance(obj, Sequence) and not isinstance(obj, str):
            if len(obj) == 1:
                for r in self.get(obj=obj[0]):
                    yield r
            elif len(obj) == 0:
                raise ValueError(
                    f"Object not a dictionary or sequence of dictionaries: {obj}."
                )
            else:  # len(obj) > 1
                logger.warning(
                    "Object (of type %s is a sequence containing more than one dictionary. "
                    "Lookup by item or attribute `%s` is ambiguous. "
                    "Use a lookup by index(es) or slice first to disambiguate.",
                    type(obj).__name__, self.item_or_attribute
                )
                for sub_obj in obj:
                    try:
                        for r in self.get(obj=sub_obj):
                            yield r
                    except Exception:
                        pass

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
                    f"Object {repr(obj)} of type {type(obj)} does not have item or attribute {repr(self.item_or_attribute)}."
                )

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = dict()

        if isinstance(obj, Dict) and not isinstance(obj, Bunch):
            # Bunch claims to be a Dict.
            # copy
            obj = {k: v for k, v in obj.items()}
            obj[self.item_or_attribute] = val
        else:
            obj = copy(obj)  # might cause issues
            setattr(obj, self.item_or_attribute, val)

        return obj

    def __repr__(self):
        return f".{self.item_or_attribute}"


class GetSlice(Step):
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]

    # Hashable requirement
    def __hash__(self):
        return hash((self.start, self.stop, self.step))

    # Step requirement
    def get(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            lower, upper, step = slice(self.start, self.stop,
                                       self.step).indices(len(obj))
            for i in range(lower, upper, step):
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = []

        assert isinstance(obj, Sequence), "Sequence expected."

        lower, upper, step = slice(self.start, self.stop,
                                   self.step).indices(len(obj))

        # copy
        obj = list(obj)

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
    indices: Tuple[int, ...]

    # Hashable requirement
    def __hash__(self):
        return hash(self.indices)

    def __init__(self, indices: Iterable[int]):
        super().__init__(indices=tuple(indices))

    # Step requirement
    def get(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            for i in self.indices:
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = []

        assert isinstance(obj, Sequence), "Sequence expected."

        # copy
        obj = list(obj)

        for i in self.indices:
            if i >= 0:
                while len(obj) <= i:
                    obj.append(None)

            obj[i] = val

        return obj

    def __repr__(self):
        return f"[{','.join(map(str, self.indices))}]"


class GetItems(Step):
    items: Tuple[str, ...]

    # Hashable requirement
    def __hash__(self):
        return hash(self.items)

    def __init__(self, items: Iterable[str]):
        super().__init__(items=tuple(items))

    # Step requirement
    def get(self, obj: Dict[str, T]) -> Iterable[T]:
        if isinstance(obj, Dict):
            for i in self.items:
                yield obj[i]
        else:
            raise ValueError("Object is not a dictionary.")

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = dict()

        assert isinstance(obj, Dict), "Dictionary expected."

        # copy
        obj = {k: v for k, v in obj.items()}

        for i in self.items:
            obj[i] = val

        return obj

    def __repr__(self):
        return "[" + (','.join(f"'{i}'" for i in self.items)) + "]"


class ParseException(Exception):

    def __init__(self, exp_string: str, exp_ast: ast.AST):
        self.exp_string = exp_string
        self.exp_ast = exp_ast

    def __str__(self):
        return (
            f"Failed to parse expression `{self.exp_string}` as a `Lens`."
            f"\nAST={dump(self.exp_ast) if self.exp_ast is not None else 'AST is None'}"
        )


class Lens(pydantic.BaseModel, Sized, Hashable):
    # Not using SerialModel as we have special handling of serialization to/from
    # strings for this class which interferes with SerialModel mechanisms.
    """
    Lenses into python objects.

    !!! example
    
        ```python
        path = Lens().record[5]['somekey']

        obj = ... # some object that contains a value at `obj.record[5]['somekey]`

        value_at_path = path.get(obj) # that value

        new_obj = path.set(obj, 42) # updates the value to be 42 instead
        ```

    ## `collect` and special attributes

    Some attributes hold special meaning for lenses. Attempting to access
    them will produce a special lens instead of one that looks up that
    attribute.

    Example:
        ```python
        path = Lens().record[:]

        obj = dict(record=[1, 2, 3])

        value_at_path = path.get(obj) # generates 3 items: 1, 2, 3 (not a list)

        path_collect = path.collect()

        value_at_path = path_collect.get(obj) # generates a single item, [1, 2, 3] (a list)
        ```
        """

    path: Tuple[Step, ...]

    @pydantic.model_validator(mode='wrap')
    @classmethod
    def validate_from_string(cls, obj, handler):
        # `mode="before"` validators currently cannot return something of a type
        # different than obj. Might be a pydantic oversight/bug.

        if isinstance(obj, str):
            ret = Lens.of_string(obj)
            return ret
        elif isinstance(obj, dict):
            return handler(
                dict(path=(Step.model_validate(step) for step in obj['path']))
            )
        else:
            return handler(obj)

    @pydantic.model_serializer
    def dump_as_string(self, **kwargs):
        return str(self)

    def __init__(self, path: Optional[Iterable[Step]] = None):
        if path is None:
            path = ()

        super().__init__(path=tuple(path))

    def existing_prefix(self, obj: Any) -> Lens:
        """Get the Lens representing the longest prefix of the path that exists
        in the given object.
        """

        last_lens = Lens()
        current_lens = last_lens

        for i, step in enumerate(self.path):
            last_lens = current_lens
            current_lens = current_lens._append(step)
            if not current_lens.exists(obj):
                return last_lens

        return current_lens

    def exists(self, obj: Any) -> bool:
        """Check whether the path exists in the given object."""

        try:
            for _ in self.get(obj):
                # Check that all named values exist, not just the first one.
                pass

        except (KeyError, IndexError, ValueError):
            return False

        return True

    @staticmethod
    def of_string(s: str) -> Lens:
        """
        Convert a string representing a python expression into a Lens.
        """

        # NOTE(piotrm): we use python parser for this which means only things
        # which are valid python expressions (with additional constraints) can
        # be converted.

        if len(s) == 0:
            return Lens()

        try:
            # NOTE: "eval" here means to parse an expression, not a statement.
            # exp = parse(f"PLACEHOLDER.{s}", mode="eval")
            exp = parse(s, mode="eval")

        except SyntaxError as e:
            raise ParseException(s, None)

        if not isinstance(exp, ast.Expression):
            raise ParseException(s, exp)

        exp = exp.body

        path = []

        def of_index(idx):

            if isinstance(idx, ast.Tuple):

                elts = tuple(of_index(elt) for elt in idx.elts)

                if all(isinstance(e, GetItem) for e in elts):
                    return GetItems(items=tuple(e.item for e in elts))

                elif all(isinstance(e, GetIndex) for e in elts):
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
                    step = of_index(sub.value)

                    path.append(step)

                elif isinstance(sub, ast.Slice):
                    vals: Tuple[GetIndex, ...] = tuple(
                        of_index(v) for v in (sub.lower, sub.upper, sub.step)
                    )

                    if not all(
                            e is None or isinstance(e, GetIndex) for e in vals):
                        raise ParseException(s, exp)

                    vals_indices: Tuple[Union[None, int], ...] = tuple(
                        None if e is None else e.index for e in vals
                    )

                    path.append(
                        GetSlice(
                            start=vals_indices[0],
                            stop=vals_indices[1],
                            step=vals_indices[2]
                        )
                    )

                elif isinstance(sub, ast.Tuple):
                    path.append(of_index(sub))

                elif isinstance(sub, ast.Constant):
                    path.append(of_index(sub))

                else:
                    raise ParseException(s, exp)

                exp = exp.value

            elif isinstance(exp, ast.List):
                # Need this case for paths that do not start with item/attribute
                # but instead go directly to something bracketed, which is a
                # list in python syntax.

                if not all(isinstance(el, ast.Constant) for el in exp.elts):
                    raise ParseException(s, exp)

                if len(exp.elts) == 0:
                    logger.warning(
                        f"Path {s} is getting zero items/indices, it will not produce anything."
                    )
                    path.append(GetIndices(indices=()))

                elif len(exp.elts) == 1:
                    el = exp.elts[0]
                    if isinstance(el.value, int):
                        path.append(GetIndex(index=el.value))
                    elif isinstance(el.value, str):
                        path.append(GetItem(item=el.value))
                    else:
                        raise ParseException(s, exp)

                else:
                    if all(isinstance(el.value, int) for el in exp.elts):
                        path.append(
                            GetIndices(
                                indices=tuple(el.value for el in exp.elts)
                            )
                        )

                    elif all(isinstance(el.value, str) for el in exp.elts):
                        path.append(
                            GetItems(items=tuple(el.value for el in exp.elts))
                        )

                    else:
                        raise ParseException(s, exp)

                exp = None

            elif isinstance(exp, ast.Name):
                path.append(GetItemOrAttribute(item_or_attribute=exp.id))

                exp = None

            elif isinstance(exp, ast.Call):
                if not isinstance(exp.func, ast.Attribute):
                    raise ParseException(s, exp)

                funcname = exp.func.attr
                if funcname == "collect":
                    path.append(Collect())

                else:
                    raise TypeError(
                        f"`{funcname}` is not a handled call for paths."
                    )

                if len(exp.args) + len(exp.keywords) != 0:
                    logger.warning(f"args/kwargs for `{funcname}` are ignored")

                exp = exp.func.value

            else:
                raise ParseException(s, exp)

        return Lens(path=path[::-1])

    def __str__(self):
        ret = ""
        for step in self.path:
            if isinstance(step, StepItemOrAttribute) and ret == "":
                ret = step.get_item_or_attribute()
            else:
                ret += repr(step)

        return ret

    def __repr__(self):
        return "Lens()" + ("".join(map(repr, self.path)))

    # Hashable requirement
    def __hash__(self):
        return hash(self.path)

    # Sized requirement
    def __len__(self):
        return len(self.path)

    def __add__(self, other: 'Lens'):
        return Lens(path=self.path + other.path)

    def is_immediate_prefix_of(self, other: 'Lens'):
        return self.is_prefix_of(other) and len(self.path) + 1 == len(
            other.path
        )

    def is_prefix_of(self, other: 'Lens'):
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

    def set(self, obj: T, val: Union[Any, T]) -> T:
        """
        In `obj` at path `self` exists, change it to `val`. Otherwise create a
        spot for it with Munch objects and then set it.
        """

        if len(self.path) == 0:
            return val

        first = self.path[0]
        rest = Lens(path=self.path[1:])

        try:
            firsts = first.get(obj)
            first_obj, firsts = iterable_peek(firsts)

        except (ValueError, IndexError, KeyError, AttributeError):

            # `first` points to an element that does not exist, use `set` to create a spot for it.
            obj = first.set(obj, None)  # will create a spot for `first`
            firsts = first.get(obj)

        for first_obj in firsts:
            obj = first.set(obj, rest.set(first_obj, val))

        return obj

    def get_sole_item(self, obj: Any) -> Any:
        all_objects = list(self.get(obj))

        assert len(
            all_objects
        ) == 1, f"Lens {self} did not address exactly a single object."

        return all_objects[0]

    def __call__(self, *args, **kwargs):
        error_msg = (
            "Only `collect` is a valid function name in python call syntax when building lenses. "
            "Note that applying a selector/path/JSONPAth/Lens now requires to use the `get` method instead of `__call__`."
        )

        assert len(self.path) > 0, error_msg
        assert isinstance(self.path[-1], StepItemOrAttribute), error_msg

        funcname = self.path[-1].get_item_or_attribute()

        if funcname == "collect":
            return Lens(path=self.path[0:-1] + (Collect(),))

        else:
            raise TypeError(error_msg)

    def get(self, obj: Any) -> Iterable[Any]:
        if len(self.path) == 0:
            yield obj
            return

        # Need to do the recursion by breaking down steps from the end in order
        # to support `Collect`.
        last_step = self.path[-1]
        if len(self.path) == 1:
            start = Lens(path=())
        else:
            start = Lens(path=self.path[0:-1])

        start_items = start.get(obj)

        if isinstance(last_step, Collect):
            yield list(start_items)

        else:
            for start_selection in start_items:
                for last_selection in last_step.get(start_selection):
                    yield last_selection

    def _append(self, step: Step) -> Lens:
        return Lens(path=self.path + (step,))

    def __getitem__(
        self, item: Union[int, str, slice, Sequence[int], Sequence[str]]
    ) -> 'Lens':
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

    def __getattr__(self, attr: str) -> 'Lens':
        if attr == "_ipython_canary_method_should_not_exist_":
            # NOTE(piotrm): when displaying objects, ipython checks whether they
            # have overwritten __getattr__ by looking up this attribute. If it
            # does not result in AttributeError or None, IPython knows it was
            # overwritten and it will not try to use any of the _repr_*_ methods
            # to display the object. In our case, this will result Lenses being
            # constructed with this canary attribute name. We instead return
            # None here to let ipython know we have overwritten __getattr__ but
            # we do not construct any Lenses.
            return 0xdead

        return self._append(GetItemOrAttribute(item_or_attribute=attr))


Lens.model_rebuild()

# TODO: Deprecate old name.
JSONPath = Lens


def leaf_queries(obj_json: JSON, query: Lens = None) -> Iterable[Lens]:
    """
    Get all queries for the given object that select all of its leaf values.
    """

    query = query or Lens()

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


def all_queries(obj: Any, query: Lens = None) -> Iterable[Lens]:
    """
    Get all queries for the given object.
    """

    query = query or Lens()

    if isinstance(obj, JSON_BASES):
        yield query

    elif isinstance(obj, pydantic.BaseModel):
        yield query

        for k in obj.model_fields:
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


def all_objects(obj: Any, query: Lens = None) -> Iterable[Tuple[Lens, Any]]:
    """
    Get all queries for the given object.
    """

    query = query or Lens()

    yield (query, obj)

    if isinstance(obj, JSON_BASES):
        pass

    elif isinstance(obj, pydantic.BaseModel):
        for k in obj.model_fields:
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
        logger.debug(
            f"Cannot create query for Iterable types like {obj.__class__.__name__} at query {query}. Convert the iterable to a sequence first."
        )

    else:
        logger.debug(f"Unhandled object type {obj} {type(obj)}")


def leafs(obj: Any) -> Iterable[Tuple[str, Any]]:
    for q in leaf_queries(obj):
        path_str = str(q)
        val = q(obj)
        yield (path_str, val)


def matching_objects(obj: Any, match: Callable) -> Iterable[Tuple[Lens, Any]]:
    for q, val in all_objects(obj):
        if match(q, val):
            yield (q, val)


def matching_queries(obj: Any, match: Callable) -> Iterable[Lens]:
    for q, _ in matching_objects(obj, match=match):
        yield q


# HACK013:
SerialModel.model_rebuild()
SerialBytes.model_rebuild()
Step.model_rebuild()
Collect.model_rebuild()
StepItemOrAttribute.model_rebuild()
GetAttribute.model_rebuild()
GetIndex.model_rebuild()
GetItem.model_rebuild()
GetItemOrAttribute.model_rebuild()
GetSlice.model_rebuild()
GetIndices.model_rebuild()
GetItems.model_rebuild()
Lens.model_rebuild()

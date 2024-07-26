"""
Serialization Typing utilities.

This should not depend on any other part of the TruLens codebase.
"""

from __future__ import annotations

from copy import copy
import logging
from typing import (
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from munch import Munch as Bunch
import pydantic
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema
from pydantic_core import core_schema

logger = logging.getLogger(__name__)

T = TypeVar("T")

JSON_BASES: Tuple[type, ...] = (str, int, float, bytes, type(None))
"""
Tuple of JSON-able base types.

Can be used in `isinstance` checks.
"""

JSON_BASES_T = Union[str, int, float, bytes, None]
"""
Alias for JSON-able base types.
"""

JSON = Union[JSON_BASES_T, Sequence[Any], Dict[str, Any]]
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


# NOTE1: Lens, a container for selector/accessors/setters of data stored in a
# json structure. Cannot make abstract since pydantic will try to initialize it.
class Step(pydantic.BaseModel, Hashable):
    """
    A step in a selection path.
    """

    def __hash__(self) -> int:
        raise TypeError(f"Should never be called, self={self.model_dump()}")

    @classmethod
    def model_validate(cls, obj: Union[Step, Dict], **kwargs) -> Step:
        if isinstance(obj, Step):
            return super().model_validate(obj, **kwargs)

        elif isinstance(obj, dict):
            ATTRIBUTE_TYPE_MAP = {
                "item": GetItem,
                "index": GetIndex,
                "attribute": GetAttribute,
                "item_or_attribute": GetItemOrAttribute,
                "start": GetSlice,
                "stop": GetSlice,
                "step": GetSlice,
                "items": GetItems,
                "indices": GetIndices,
                "collect": Collect,
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

    def __repr__(self) -> str:
        return ".collect()"


class StepItemOrAttribute(Step):
    # NOTE1
    def get_item_or_attribute(self):
        raise NotImplementedError()


class GetAttribute(StepItemOrAttribute):
    attribute: str

    # Hashable requirement.
    def __hash__(self) -> int:
        return hash(self.attribute)

    # StepItemOrAttribute requirement
    def get_item_or_attribute(self) -> str:
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

    def __repr__(self) -> str:
        return f".{self.attribute}"


class GetIndex(Step):
    index: int

    # Hashable requirement
    def __hash__(self) -> int:
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

    def __repr__(self) -> str:
        return f"[{self.index}]"


class GetItem(StepItemOrAttribute):
    item: str

    # Hashable requirement
    def __hash__(self) -> int:
        return hash(self.item)

    # StepItemOrAttribute requirement
    def get_item_or_attribute(self) -> str:
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

    def __repr__(self) -> str:
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
    def __hash__(self) -> int:
        return hash(self.item_or_attribute)

    # StepItemOrAttribute requirement
    def get_item_or_attribute(self) -> str:
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
                    type(obj).__name__,
                    self.item_or_attribute,
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

    def __repr__(self) -> str:
        return f".{self.item_or_attribute}"


class GetSlice(Step):
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]

    # Hashable requirement
    def __hash__(self) -> int:
        return hash((self.start, self.stop, self.step))

    # Step requirement
    def get(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            lower, upper, step = slice(
                self.start, self.stop, self.step
            ).indices(len(obj))
            for i in range(lower, upper, step):
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    # Step requirement
    def set(self, obj: Any, val: Any) -> Any:
        if obj is None:
            obj = []

        assert isinstance(obj, Sequence), "Sequence expected."

        lower, upper, step = slice(self.start, self.stop, self.step).indices(
            len(obj)
        )

        # copy
        obj = list(obj)

        for i in range(lower, upper, step):
            obj[i] = val

        return obj

    def __repr__(self) -> str:
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
    def __hash__(self) -> int:
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

    def __repr__(self) -> str:
        return f"[{','.join(map(str, self.indices))}]"


class GetItems(Step):
    items: Tuple[str, ...]

    # Hashable requirement
    def __hash__(self) -> str:
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

    def __repr__(self) -> str:
        return "[" + (",".join(f"'{i}'" for i in self.items)) + "]"


# HACK013:
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

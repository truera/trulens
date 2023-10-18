"""
Serialization utilities.

TODO: Lens class: can we store just the python AST instead of building up our
own "Step" classes to hold the same data? We are already using AST for parsing.
"""

from __future__ import annotations

import ast
from ast import dump
from ast import parse
from copy import copy
import logging
from pprint import PrettyPrinter

from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple,
    TypeVar, Union
)

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
    def validate(cls, obj: Any):
        # import hierarchy circle here
        from trulens_eval.utils.pyschema import Class
        from trulens_eval.utils.pyschema import CLASS_INFO
        from trulens_eval.utils.pyschema import WithClassInfo

        if isinstance(obj, Dict):
            if CLASS_INFO in obj:

                cls = Class(**obj[CLASS_INFO])
                del obj[CLASS_INFO]
                model = cls.validate(obj=obj)

                return WithClassInfo.of_model(model=model, cls=cls)

        return super().validate(obj)
            

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

    def parse_raw(cls, data: Union[str, bytes]) -> 'SerialBytes':
        import base64

        if isinstance(data, bytes):
            return cls.parse_obj(dict(data=base64.b64decode(data)))
        else:
            return super().parse_raw(data)

    """
    @classmethod
    def parse_obj(cls, obj: Any):
        import base64

        if isinstance(obj, Dict):
            encoded = obj['data']
            return SerialBytes(data=base64.b64decode(encoded))
        else:
            raise ValueError(obj)
    """


# Lens, a container for selector/accessors/setters of data stored in a json
# structure. Cannot make abstract since pydantic will try to initialize it.
class Step(pydantic.BaseModel):  #, abc.ABC):
    """
    A step in a selection path.
    """

    @classmethod
    def validate(cls, obj):

        if isinstance(obj, Step):
            return obj
        
        elif isinstance(obj, Dict):
        
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
            
        raise RuntimeError(f"Do not know how to interpret {obj} as a `Lens` `Step`.")
            

    # @abc.abstractmethod
    def get(self, obj: Any) -> Iterable[Any]:
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

class Collect(Step):
    # Need something for `Step.validate` to tell that it is looking at Collect.
    collect: None = None
    
    def __hash__(self):
        return hash("collect")

    def get(self, obj: Any) -> Iterable[List[Any]]:
        # Needs to be handled in Lens class itself.
        raise NotImplementedError()
        
    def set(self, obj: Any, val: Any) -> Any:
        raise NotImplementedError()

    def __repr__(self):
        return f".collect()"


class StepItemOrAttribute(Step):
    def get_item_or_attribute(self):
        raise NotImplementedError()


class GetAttribute(StepItemOrAttribute):
    attribute: str

    def __hash__(self):
        return hash(self.attribute)

    def get_item_or_attribute(self):
        return self.attribute

    def get(self, obj: Any) -> Iterable[Any]:
        if hasattr(obj, self.attribute):
            yield getattr(obj, self.attribute)
        else:
            raise ValueError(
                f"Object {obj} does not have attribute: {self.attribute}"
            )

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

    def __hash__(self):
        return hash(self.index)

    def get(self, obj: Sequence[T]) -> Iterable[T]:
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

    def __hash__(self):
        return hash(self.item)

    def get_item_or_attribute(self):
        return self.item

    def get(self, obj: Dict[str, T]) -> Iterable[T]:
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

        # copy
        obj = {k: v for k, v in obj.items()}

        obj[self.item] = val
        return obj

    def __repr__(self):
        return f"[{repr(self.item)}]"


class GetItemOrAttribute(StepItemOrAttribute):
    # For item/attribute agnostic addressing.

    # NOTE: We also allow to lookup elements within sequences if the subelements
    # have the item or attribute. We issue warning if this is ambiguous (looking
    # up in a sequence of more than 1 element).

    item_or_attribute: str  # distinct from "item" for deserialization

    def __hash__(self):
        return hash(self.item_or_attribute)

    def get_item_or_attribute(self):
        return self.item_or_attribute

    def get(self, obj: Dict[str, T]) -> Iterable[T]:
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

        if isinstance(obj, Dict) and not isinstance(obj, Bunch):
            # Bunch claims to be a Dict.
            # copy
            obj = {k: v for k, v in obj.items()}
            obj[self.item_or_attribute] = val
        else:
            obj = copy(obj) # might cause issues
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

    def get(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            lower, upper, step = slice(self.start, self.stop,
                                       self.step).indices(len(obj))
            for i in range(lower, upper, step):
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    def set(self, obj: Any, val: Any) -> Any:
        # raise NotImplementedError
    
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
    indices: Sequence[int]

    def __hash__(self):
        return hash(tuple(self.indices))

    def __init__(self, indices):
        super().__init__(indices=tuple(indices))

    def get(self, obj: Sequence[T]) -> Iterable[T]:
        if isinstance(obj, Sequence):
            for i in self.indices:
                yield obj[i]
        else:
            raise ValueError("Object is not a sequence.")

    def set(self, obj: Any, val: Any) -> Any:
        # raise NotImplementedError
        
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
    items: Sequence[str]

    def __hash__(self):
        return hash(tuple(self.items))

    def __init__(self, items):
        super().__init__(items=tuple(items))

    def get(self, obj: Dict[str, T]) -> Iterable[T]:
        if isinstance(obj, Dict):
            for i in self.items:
                yield obj[i]
        else:
            raise ValueError("Object is not a dictionary.")

    def set(self, obj: Any, val: Any) -> Any:
        # raise NotImplementedError

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
        return f"Failed to parse expression `{self.exp_string}` as a `Lens`.\nAST={dump(self.exp_ast) if self.exp_ast is not None else 'AST is None'}"

class Lens(pydantic.BaseModel):
    # Not using SerialModel as we have special handling of serialization to/from
    # strings for this class which interferes with SerialModel mechanisms.

    """
    Lenses into python objects.

    **Usage:**
    
    ```python

        path = Lens().record[5]['somekey']

        obj = ... # some object that contains a value at `obj.record[5]['somekey]`

        value_at_path = path.get(obj) # that value

        new_obj = path.set(obj, 42) # updates the value to be 42 instead
    ```
    """

    path: Tuple[Step, ...]

    @classmethod
    def validate(cls, obj):
        if isinstance(obj, str):
            return Lens.of_string(obj)
        else:
            return super().validate(obj)

    def json(self):
        return str(self)
    
    def __init__(self, path: Optional[Tuple[Step, ...]] = None):
        if path is None:
            path = ()
            
        super().__init__(path=path)
    

    @staticmethod
    def of_string(s: str) -> 'Lens':
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
                    logger.warning(f"Path {s} is getting zero items/indices, it will not produce anything.")
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
                        path.append(GetIndices(indices=tuple(el.value for el in exp.elts)))

                    elif all(isinstance(el.value, str) for el in exp.elts):
                        path.append(GetItems(items=tuple(el.value for el in exp.elts)))
                    
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
                    path.append(
                        Collect()
                    )

                else:
                    raise TypeError(f"`{funcname}` is not a handled call for paths.")

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

    def __hash__(self):
        return hash(self.path)

    def __len__(self):
        return len(self.path)

    def __add__(self, other: Lens):
        return Lens(path=self.path + other.path)

    def is_immediate_prefix_of(self, other: Lens):
        return self.is_prefix_of(other) and len(self.path) + 1 == len(
            other.path
        )

    def is_prefix_of(self, other: Lens):
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
            obj = first.set(
                obj,
                rest.set(first_obj, val)
            )

        return obj

    def get_sole_item(self, obj: Any) -> Any:
        all_objects = list(self.get(obj))

        assert len(all_objects) == 1, f"Lens {self} did not address exactly a single object."

        return all_objects[0]

    def __call__(self, *args, **kwargs):
        error_msg = ("Only `collect` is a valid function name in python call syntax when building lenses. "
                     "Note that applying a selector/path/JSONPAth/Lens now requires to use the `get` method instead of `__call__`.")

        assert len(self.path) > 0, error_msg
        assert isinstance(self.path[-1], StepItemOrAttribute), error_msg

        funcname = self.path[-1].get_item_or_attribute()

        if funcname == "collect":
            return Lens(path=self.path[0:-1] + (Collect(), ))

        else:
            raise TypeError(error_msg)

    # TODO: reserving __call__ for a different purposes than get

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
        self, item: int | str | slice | Sequence[int] | Sequence[str]
    ) -> Lens:
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

    def __getattr__(self, attr: str) -> Lens:
        if attr == "_ipython_canary_method_should_not_exist_":
            # NOTE(piotrm): when displaying objects, ipython checks whether they
            # have overwritten __getattr__ by looking up this attribute. If it
            # does not result in AttributeError or None, IPython knows it was
            # overwritten and it will not try to use any of the _repr_*_ methdos
            # to display the object. In our case, this will result Lenses being
            # constructed with this canary attribute name. We instead return
            # None here to let ipython know we have overwritten __getattr__ but
            # we do not construct any Lenses.
            return 0xdead
        
        return self._append(GetItemOrAttribute(item_or_attribute=attr))

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


def all_objects(
    obj: Any,
    query: Lens = None
) -> Iterable[Tuple[Lens, Any]]:
    """
    Get all queries for the given object.
    """

    query = query or Lens()

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
        logger.debug(f"Cannot create query for Iterable types like {obj.__class__.__name__} at query {query}. Convert the iterable to a sequence first.")

    else:
        logger.debug(f"Unhandled object type {obj} {type(obj)}")


def leafs(obj: Any) -> Iterable[Tuple[str, Any]]:
    for q in leaf_queries(obj):
        path_str = str(q)
        val = q(obj)
        yield (path_str, val)


def matching_objects(obj: Any,
                     match: Callable) -> Iterable[Tuple[Lens, Any]]:
    for q, val in all_objects(obj):
        if match(q, val):
            yield (q, val)


def matching_queries(obj: Any, match: Callable) -> Iterable[Lens]:
    for q, _ in matching_objects(obj, match=match):
        yield q

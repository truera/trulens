"""
Container class utilities.
"""

from __future__ import annotations

import datetime
import itertools
import logging
from pprint import PrettyPrinter
from typing import Callable, Dict, Iterable, Sequence, Tuple, TypeVar, Union
from pydantic.json_schema import JsonSchemaValue

import pandas as pd

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")

# Time container utilities

def datetime_of_ns_timestamp(timestamp: int) -> datetime.datetime:
    """Convert a nanosecond timestamp to a datetime."""
    return pd.Timestamp(timestamp, unit='ns').to_pydatetime()

def ns_timestamp_of_datetime(dt: datetime.datetime) -> int:
    """Convert a datetime to a nanosecond timestamp."""
    return pd.Timestamp(dt).as_unit('ns').value

# Dicts utilities

class DictNamespace(Dict[str, T]):
    """View into a dict with keys prefixed by some `namespace` string.
    
    Replicates the values without the prefix in self.
    """

    def __init__(self, parent: Dict[str, T], namespace: str, **kwargs):
        self.parent = parent
        self.namespace = namespace

    def __getitem__(self, key: str) -> T:
        return dict.__getitem__(self, key)

    def __setitem__(self, key: str, value: T) -> None:
        dict.__setitem__(self, key, value)
        self.parent[f"{self.namespace}.{key}"] = value

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self, key)
        del self.parent[f"{self.namespace}.{key}"]

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema, _handler) -> JsonSchemaValue:
        return { 
            'description': 'View into a dict with keys prefixed by some `namespace` string.',
            'title': 'DictNamespace',
            'type': 'object'
        }

# Collection utilities


def first(seq: Sequence[T]) -> T:
    """Get the first item in a sequence."""
    return seq[0]


def second(seq: Sequence[T]) -> T:
    """Get the second item in a sequence."""
    return seq[1]


def third(seq: Sequence[T]) -> T:
    """Get the third item in a sequence."""
    return seq[2]


def is_empty(obj):
    """Check if an object is empty.
    
    If object is not a sequence, returns False.
    """

    try:
        return len(obj) == 0
    except Exception:
        return False


def dict_set_with(dict1: Dict[A, B], dict2: Dict[A, B]) -> Dict[A, B]:
    """Add the key/values from `dict2` to `dict1`.
    
    Mutates and returns `dict1`.
    """

    dict1.update(dict2)
    return dict1


def dict_set_with_multikey(
    dict1: Dict[A, B], dict2: Dict[Union[A, Tuple[A, ...]], B]
) -> Dict[A, B]:
    """Like `dict_set_with` except the second dict can have tuples as keys in which
    case all of the listed keys are set to the given value."""
    for ks, v in dict2.items():
        if isinstance(ks, tuple):
            for k in ks:
                dict1[k] = v
        else:
            dict1[ks] = v
    return dict1


def dict_merge_with(dict1: Dict, dict2: Dict, merge: Callable) -> Dict:
    """Merge values from the second dictionary into the first.
    
    If both dicts contain the same key, the given `merge` function is used to
    merge the values.
    """
    for k, v in dict2.items():
        if k in dict1:
            dict1[k] = merge(dict1[k], v)
        else:
            dict1[k] = v

    return dict1


# Generator utils


def iterable_peek(it: Iterable[T]) -> Tuple[T, Iterable[T]]:
    iterator = iter(it)
    item = next(iterator)
    return item, itertools.chain([item], iterator)

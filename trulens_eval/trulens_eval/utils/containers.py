"""
Container class utilities.
"""

from __future__ import annotations

import itertools
import logging
from pprint import PrettyPrinter
from typing import Callable, Dict, Iterable, Sequence, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")

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

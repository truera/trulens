"""
Container class utilities.
"""

from __future__ import annotations

import itertools
import logging
from pprint import PrettyPrinter
from typing import Callable, Dict, Iterable, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")

# Collection utilities


def first(seq: Sequence[T]) -> T:
    return seq[0]


def second(seq: Sequence[T]) -> T:
    return seq[1]


def third(seq: Sequence[T]) -> T:
    return seq[2]


def is_empty(obj):
    try:
        return len(obj) == 0
    except Exception:
        return False


def dict_set_with(dict1: Dict, dict2: Dict):
    dict1.update(dict2)
    return dict1


def dict_merge_with(dict1: Dict, dict2: Dict, merge: Callable) -> Dict:
    """
    Merge values from the second dictionary into the first. If both dicts
    contain the same key, the given `merge` function is used to merge the
    values.
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

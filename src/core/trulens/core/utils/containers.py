"""
Container class utilities.
"""

from __future__ import annotations

import datetime
import itertools
import logging
from pprint import PrettyPrinter
from threading import Event
from threading import RLock
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import pandas as pd

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")


def datetime_of_ns_timestamp(timestamp: int) -> datetime.datetime:
    """Convert a nanosecond timestamp to a datetime."""
    return pd.Timestamp(timestamp, unit="ns").to_pydatetime()


def ns_timestamp_of_datetime(dt: datetime.datetime) -> int:
    """Convert a datetime to a nanosecond timestamp."""
    return pd.Timestamp(dt).as_unit("ns").value


class BlockingSet(set, Generic[T]):
    """A set with max size that has blocking peek/get/add ."""

    def __init__(
        self, items: Optional[Iterable[T]] = None, max_size: int = 1024
    ):
        if items is not None:
            items = list(items)
        else:
            items = []

        if len(items) > max_size:
            raise ValueError("Initial items exceed max size.")

        self.content: Set[T] = set(items)
        self.max_size = max_size

        # TODO: unsure if these 2 locks are sufficient to prevent all deadlocks
        self.read_lock = RLock()
        self.write_lock = RLock()
        self.event_nonempty = Event()
        self.event_nonfull = Event()
        self.event_shutdown = Event()

        if len(self.content) > 0:
            self.event_nonempty.set()

        if len(self.content) < self.max_size:
            self.event_nonfull.set()

    def empty(self) -> bool:
        """Check if the set is empty."""
        return len(self.content) == 0

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown the set."""

        self.event_shutdown.set()
        self.event_nonempty.set()  # Unblock any waiting threads

    def peek(self) -> T:
        """Get an item from the set.

        Blocks until an item is available.
        """

        with self.read_lock:
            self.event_nonempty.wait()
            if self.event_shutdown.is_set():
                raise StopIteration("Set is shutdown.")

            return next(iter(self.content))

    def remove(self, item: T):
        """Remove an item from the set."""
        with self.write_lock:
            self.content.remove(item)

            self.event_nonfull.set()

            if len(self.content) == 0:
                self.event_nonempty.clear()

    def pop(self, blocking: bool = True) -> Optional[T]:
        """Get and remove an item from the set.

        Blocks until an item is available, unless blocking is set to False.

        Args:
            blocking: Whether to block until an item is ready. If not blocking
                and empty, will return None.
        """

        if self.read_lock.acquire(blocking=blocking):
            if not blocking and not self.event_nonempty.is_set():
                self.read_lock.release()
                return None

            self.event_nonempty.wait()

            if self.event_shutdown.is_set():
                self.read_lock.release()
                raise StopIteration("Set is shutdown.")

            item = next(iter(self.content))
            self.content.remove(item)

            if len(self.content) == 0:
                self.event_nonempty.clear()

            self.read_lock.release()

        else:
            # Acquire was used with blocking=False and it would have
            # blocked. Want the non-blocking behaviour to return None in such cases.
            return None

        return item

    def add(self, item: T):
        """Add an item to the set.

        Blocks if set is full.
        """

        with self.write_lock:
            self.event_nonfull.wait()
            self.content.add(item)

            self.event_nonempty.set()
            if len(self.content) >= self.max_size:
                self.event_nonfull.clear()


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

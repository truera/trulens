"""Python compatibility utilities.

These are mostly type/function aliases to provide consistency across python
versions.

Imports from this module can violate the import modules standard. That is, it is
ok to import the defined names directly as they are replacement to standard
python names.
"""

from __future__ import annotations

from concurrent import futures
import inspect
import queue
import sys
import types
from typing import (
    Any,
    Generic,
    TypeVar,
)
import weakref

import typing_extensions

TypeAliasType = typing_extensions.TypeAliasType
TypeAlias = typing_extensions.TypeAlias
Type = typing_extensions.Type

if sys.version_info >= (3, 11):
    getmembers_static = inspect.getmembers_static
else:

    def getmembers_static(obj, predicate=None):
        """Implementation of inspect.getmembers_static for python < 3.11."""

        if predicate is None:
            predicate = lambda name, value: True

        return [
            (name, value)
            for name in dir(obj)
            if hasattr(obj, name)
            and predicate(name, value := getattr(obj, name))
        ]


if sys.version_info >= (3, 9):
    Future = futures.Future
    """Alias for [concurrent.futures.Future][].

    In python < 3.9, a subclass of [concurrent.futures.Future][] with
    `Generic[A]` is used instead.
    """

    Queue = queue.Queue
    """Alias for [queue.Queue][] .

    In python < 3.9, a subclass of [queue.Queue][] with
    `Generic[A]` is used instead.
    """

    WeakSet = weakref.WeakSet
    """Alias for [weakref.WeakSet][] .

    In python < 3.9, a subclass of [weakref.WeakSet][] with
    `Generic[A]` is used instead.
    """

    ReferenceType = weakref.ReferenceType
    """Alias for [weakref.ReferenceType][] .

    In python < 3.9, a subclass of [weakref.ReferenceType][] with
    `Generic[A]` is used instead.
    """

else:
    # Fake classes which can have type args. In python earlier than 3.9, the
    # classes imported above cannot have type args which is annoying for type
    # annotations. We use these fake ones instead.

    A = TypeVar("A")

    # HACK011
    class Future(Generic[A], futures.Future):
        """Alias for [concurrent.futures.Future][].

        In python < 3.9, a subclass of [concurrent.futures.Future][] with
        `Generic[A]` is used instead.
        """

    # HACK012
    class Queue(Generic[A], queue.Queue):
        """Alias for [queue.Queue][] .

        In python < 3.9, a subclass of [queue.Queue][] with
        `Generic[A]` is used instead.
        """

    class WeakSet(Generic[A], weakref.WeakSet):
        """Alias for [weakref.WeakSet][] .

        In python < 3.9, a subclass of [weakref.WeakSet][] with
        `Generic[A]` is used instead.
        """

    class ReferenceType(Generic[A], weakref.ReferenceType):
        """Alias for [weakref.ReferenceType][] .

        In python < 3.9, a subclass of [weakref.ReferenceType][] with
        `Generic[A]` is used instead.
        """


class EmptyType(type):
    """A type that cannot be instantiated or subclassed."""

    def __new__(mcs, *args, **kwargs):
        raise ValueError("EmptyType cannot be instantiated.")

    def __instancecheck__(cls, __instance: Any) -> bool:
        return False

    def __subclasscheck__(cls, __subclass: Type) -> bool:
        return False


if sys.version_info >= (3, 10):
    NoneType: TypeAlias = types.NoneType
    """Alias for [types.NoneType][] .

    In python < 3.10, it is defined as `type(None)` instead.
    """

else:
    NoneType = type(None)
    """Alias for [types.NoneType][] .

    In python < 3.10, it is defined as `type(None)` instead.
    """

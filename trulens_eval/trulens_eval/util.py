"""
Utilities.

Do not import anything from trulens_eval here.
"""

from __future__ import annotations
from functools import singledispatch
import logging
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import ThreadPool
from queue import Queue
from time import sleep
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, TypeVar, Union

from multiprocessing.context import TimeoutError
from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

T = TypeVar("T")

UNICODE_CHECK = "✅"
UNCIODE_YIELD = "⚡"


def first(seq: Sequence[T]) -> T:
    return seq[0]


def second(seq: Sequence[T]) -> T:
    return seq[1]


def third(seq: Sequence[T]) -> T:
    return seq[2]


class JSONPath(object):
    # TODO(piotrm): more appropriate version of tinydb.Query

    class Step():
        pass

    @dataclass
    class GetAttribute(Step):
        attribute: str

        def __call__(self, obj: Any) -> Iterable[Any]:
            if hasattr(obj, self.attribute):
                yield getattr(self, self.attribute)
            else:
                raise ValueError(f"Object does not have attribute: {self.attribute}")

    @dataclass
    class GetIndex(Step):
        index: int

        def __call__(self, obj: Sequence[T]) -> Iterable[T]: 
            if isinstance(obj, Sequence):
                if len(obj) > self.index:
                    yield obj[self.index]
                else:
                    raise IndexError(f"Index out of bounds: {self.index}")
            else:
                raise ValueError("Object is not a sequence.")

    @dataclass
    class GetItem(Step):
        item: str

        def __call__(self, obj: Dict[str, T]) -> Iterable[T]:
            if isinstance(obj, Dict):
                if self.item in obj:
                    yield obj[self.item]
                else:
                    raise KeyError(f"Key not in dictionary: {self.item}")
            else:
                raise ValueError("Object is not a dictionary.")

    @dataclass
    class GetSlice(Step):
        slice: slice

        def __call__(self, obj: Sequence[T]) -> Iterable[T]:
            if isinstance(obj, Sequence):
                lower, upper, step = self.slice.indices(len(obj))
                for i in range(lower, upper, step):
                    yield obj[i]
            else:
                raise ValueError("Object is not a sequence.")

    @dataclass
    class GetIndices(Step):
        indices: Sequence[int]

        def __call__(self, obj: Sequence[T]) -> Iterable[T]:
            if isinstance(obj, Sequence):
                for i in self.indices:
                    yield obj[i]
            else:
                raise ValueError("Object is not a sequence.")


    @dataclass
    class GetItems(Step):
        items: Sequence[str]

        def __call__(self, obj: Dict[T]) -> Iterable[T]:
            if isinstance(obj, Dict):
                for i in self.items:
                    yield obj[i]
            else:
                raise ValueError("Object is not a dictionary.")


    class Aggregate(Step):
        pass

    @property
    def path(self):
        return self._path

    def __init__(self, path=None):
        self._path = path or []

    def __str__(self):
        return "JSONPath(" + ".".join(map(str, self._path)) + ")"

    def __repr__(self):
        return str(self)

    def __call__(self, obj: Any) -> Iterable[Any]:
        if len(self._path) == 0:
            yield obj
            return
        
        first = self._path[0]
        if len(self._path) == 1:
            rest = JSONPath()
        else:
            rest = JSONPath(self._path[1:])

        for first_selection in first.__call__(obj):
            for rest_selection in rest.__call__(first_selection):
                yield rest_selection

    def __agg(self, aggregator: str) -> Any:
        pass

    def __append(self, step: JSONPath.Step) -> JSONPath:
        return JSONPath(self._path + [step])

    def __getitem__(
        self, item: int | str | slice | Sequence[int] | Sequence[str]
    ) -> JSONPath:
        if isinstance(item, int):
            return self.__append(JSONPath.GetIndex(item))
        if isinstance(item, str):
            return self.__append(JSONPath.GetItem(item))
        if isinstance(item, slice):
            return self.__append(JSONPath.GetSlice(item))
        if isinstance(item, Sequence):
            item = tuple(item)
            if all(isinstance(i, int) for i in item):
                return self.__append(JSONPath.GetIndices(item))
            elif all(isinstance(i, str) for i in item):
                return self.__append(JSONPath.GetItems(item))
            else:
                raise TypeError(
                    f"Unhandled sequence item types: {list(map(type, item))}. "
                    f"Note mixing int and str is not allowed."
                )

        raise TypeError(f"Unhandled item type {type(item)}.")

    def __getattr__(self, attr: str) -> JSONPath:
        return self.__append(JSONPath.GetAttribute(attr))


class SingletonPerName():
    """
    Class for creating singleton instances except there being one instance max,
    there is one max per different `name` argument. If `name` is never given,
    reverts to normal singleton behaviour.
    """

    # Hold singleton instances here.
    instances: Dict[Hashable, 'SingletonPerName'] = dict()

    def __new__(cls, name: str = None, *args, **kwargs):
        """
        Create the singleton instance if it doesn't already exist and return it.
        """

        key = cls.__name__, name

        if key not in cls.instances:
            logging.debug(
                f"*** Creating new {cls.__name__} singleton instance for name = {name} ***"
            )
            SingletonPerName.instances[key] = super().__new__(cls)

        return SingletonPerName.instances[key]


class TP(SingletonPerName):  # "thread processing"

    def __init__(self):
        if hasattr(self, "thread_pool"):
            # Already initialized as per SingletonPerName mechanism.
            return

        # TODO(piotrm): if more tasks than `processes` get added, future ones
        # will block and earlier ones may never start executing.
        self.thread_pool = ThreadPool(processes=1024)
        self.running = 0
        self.promises = Queue(maxsize=1024)

    def runrepeatedly(self, func: Callable, rpm: float = 6, *args, **kwargs):

        def runner():
            while True:
                func(*args, **kwargs)
                sleep(60 / rpm)

        self.runlater(runner)

    def runlater(self, func: Callable, *args, **kwargs) -> None:
        prom = self.thread_pool.apply_async(func, args=args, kwds=kwargs)
        self.promises.put(prom)

    def promise(self, func: Callable[..., T], *args, **kwargs) -> AsyncResult:
        prom = self.thread_pool.apply_async(func, args=args, kwds=kwargs)
        self.promises.put(prom)

        return prom

    def finish(self, timeout: Optional[float] = None) -> int:
        print(f"Finishing {self.promises.qsize()} task(s) ", end='')

        timeouts = []

        while not self.promises.empty():
            prom = self.promises.get()
            try:
                prom.get(timeout=timeout)
                print(".", end="")
            except TimeoutError:
                print("!", end="")
                timeouts.append(prom)

        for prom in timeouts:
            self.promises.put(prom)

        if len(timeouts) == 0:
            print("done.")
        else:
            print("some tasks timed out.")

        return len(timeouts)

    def _status(self) -> List[str]:
        rows = []

        for p in self.thread_pool._pool:
            rows.append([p.is_alive(), str(p)])

        return pd.DataFrame(rows, columns=["alive", "thread"])

"""
Utilities.

Do not import anything from trulens_eval here.
"""

import logging
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import ThreadPool
from typing import Callable, Dict, Hashable, List, TypeVar

import pandas as pd

T = TypeVar("T")


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

        if name not in cls.instances:
            logging.debug(
                f"creating {cls} singleton instance for name = {name}"
            )
            cls.instances[name] = super().__new__(cls)

        return cls.instances[name]


class TP(SingletonPerName):  # "thread processing"

    def __init__(self):
        self.thread_pool = ThreadPool(processes=8)
        self.running = 0

    def _started(self, *args, **kwargs):
        print("started async call")
        self.running += 1

    def _finished(self, *args, **kwargs):
        print("finished async call")
        self.running -= 1

    def runlater(self, func: Callable, *args, **kwargs) -> None:
        self._started()
        self.thread_pool.apply_async(func, callback=self._finished, args=args, kwds=kwargs)

    def promise(self, func: Callable[..., T], *args,
                **kwargs) -> AsyncResult[T]:
        self._started()
        return self.thread_pool.apply_async(func, callback=self._finished, args=args, kwds=kwargs)
    
    def status(self) -> List[str]:
        rows = []

        for p in self.thread_pool._pool:
            rows.append([p.is_alive(), str(p)])
            
        return pd.DataFrame(rows, columns=["alive", "thread"])

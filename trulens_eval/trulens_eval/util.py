"""
Utilities.

Do not import anything from trulens_eval here.
"""

import logging
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import ThreadPool
from time import sleep
from typing import Callable, Dict, Hashable, List, TypeVar

import pandas as pd
from tqdm.auto import tqdm

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

        key = cls.__name__, name

        if key not in cls.instances:
            #logging.debug(
            print(
                f"*** Creating new {cls.__name__} singleton instance for name = {name} ***"
            )
            SingletonPerName.instances[key] = super().__new__(cls)

        return SingletonPerName.instances[key]


class TP(SingletonPerName):  # "thread processing"

    def __init__(self):
        if hasattr(self, "thread_pool"):
            # Already initialized as per SingletonPerName mechanism.
            return

        self.thread_pool = ThreadPool(processes=16)
        self.running = 0

    def _started(self, *args, **kwargs):
        self.running += 1

    def _finished(self, *args, **kwargs):
        self.running -= 1

    def runrepeatedly(self, func: Callable, rpm: float = 6, *args, **kwargs):
        def runner():
            while True:
                func(*args, **kwargs)
                sleep(60 / rpm)

        self.runlater(runner)

    def runlater(self, func: Callable, *args, **kwargs) -> None:
        self._started()
        self.thread_pool.apply_async(func, callback=self._finished, args=args, kwds=kwargs)

    def promise(self, func: Callable[..., T], *args,
                **kwargs) -> AsyncResult:
        self._started()
        return self.thread_pool.apply_async(func, callback=self._finished, args=args, kwds=kwargs)
    
    def status(self) -> List[str]:
        rows = []

        for p in self.thread_pool._pool:
            rows.append([p.is_alive(), str(p)])
            
        return pd.DataFrame(rows, columns=["alive", "thread"])

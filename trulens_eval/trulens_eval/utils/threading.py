"""
Multi-threading utilities.
"""

from concurrent.futures import ThreadPoolExecutor as fThreadPoolExecutor
from inspect import stack
import logging
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import ThreadPool
from queue import Queue
from time import sleep
from typing import Callable, List, Optional, TypeVar

import pandas as pd

from trulens_eval.utils.python import _future_target_wrapper
from trulens_eval.utils.python import SingletonPerName

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ThreadPoolExecutor(fThreadPoolExecutor):

    def submit(self, fn, /, *args, **kwargs):
        present_stack = stack()
        return super().submit(
            _future_target_wrapper, present_stack, fn, *args, **kwargs
        )


class TP(SingletonPerName):  # "thread processing"

    # Store here stacks of calls to various thread starting methods so that we
    # can retrieve the trace of calls that caused a thread to start.

    # pre_run_stacks = dict()

    def __init__(self):
        if hasattr(self, "thread_pool"):
            # Already initialized as per SingletonPerName mechanism.
            return

        # TODO(piotrm): if more tasks than `processes` get added, future ones
        # will block and earlier ones may never start executing.
        self.thread_pool = ThreadPool(processes=64)
        self.running = 0
        self.promises = Queue(maxsize=64)

    def runrepeatedly(self, func: Callable, rpm: float = 6, *args, **kwargs):

        def runner():
            while True:
                func(*args, **kwargs)
                sleep(60 / rpm)

        self.runlater(runner)

    def _thread_starter(self, func, args, kwargs):
        present_stack = stack()

        prom = self.thread_pool.apply_async(
            _future_target_wrapper,
            args=(present_stack, func) + args,
            kwds=kwargs
        )

        return prom

    def finish_if_full(self):
        if self.promises.full():
            print("Task queue full. Finishing existing tasks.")
            self.finish()

    def runlater(self, func: Callable, *args, **kwargs) -> None:
        self.finish_if_full()

        prom = self._thread_starter(func, args, kwargs)

        self.promises.put(prom)

    def promise(self, func: Callable[..., T], *args, **kwargs) -> AsyncResult:
        self.finish_if_full()

        prom = self._thread_starter(func, args, kwargs)

        self.promises.put(prom)

        return prom

    def finish(self, timeout: Optional[float] = None) -> int:
        logger.debug(f"Finishing {self.promises.qsize()} task(s).")

        timeouts = []

        while not self.promises.empty():
            prom = self.promises.get()
            try:
                prom.get(timeout=timeout)
            except TimeoutError:
                timeouts.append(prom)

        for prom in timeouts:
            self.promises.put(prom)

        if len(timeouts) == 0:
            logger.debug("Done.")
        else:
            logger.debug("Some tasks timed out.")

        return len(timeouts)

    def _status(self) -> List[str]:
        rows = []

        for p in self.thread_pool._pool:
            rows.append([p.is_alive(), str(p)])

        return pd.DataFrame(rows, columns=["alive", "thread"])

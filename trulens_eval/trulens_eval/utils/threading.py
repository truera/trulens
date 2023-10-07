"""
Multi-threading utilities.
"""

from concurrent.futures import Future, as_completed, wait
from concurrent.futures import ThreadPoolExecutor as fThreadPoolExecutor
from concurrent.futures import TimeoutError
from inspect import stack
import logging
from queue import Queue
from threading import Lock, Thread
import threading
from time import sleep
from typing import Callable, Optional, TypeVar
import warnings

from trulens_eval.utils.python import _future_target_wrapper
from trulens_eval.utils.python import SingletonPerName

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ThreadPoolExecutor(fThreadPoolExecutor):
    """
    A ThreadPoolExecutor that keeps track of the stack prior to each thread's
    invocation.
    """

    def submit(self, fn, /, *args, **kwargs):
        present_stack = stack()
        return super().submit(
            _future_target_wrapper, present_stack, fn, *args, **kwargs
        )


class TP(SingletonPerName):  # "thread processing"

    # Store here stacks of calls to various thread starting methods so that we
    # can retrieve the trace of calls that caused a thread to start.

    MAX_THREADS = 128

    # How long to wait for any task before restarting it.
    ROBUST_TIMEOUT = 60.0

    # How many times to restart a failed or timed-out task.
    ROBUST_RETRIES = 3

    def __init__(self):
        if hasattr(self, "thread_pool"):
            # Already initialized as per SingletonPerName mechanism.
            return

        # Run tasks started with this class using this pool.
        self.thread_pool = fThreadPoolExecutor(max_workers=TP.MAX_THREADS, thread_name_prefix="TP.submit")

        # Store the futures for the tasks started with this class here.
        # This enforces an upper bound on how many tasks can be queued at once.
        # self.futures = Queue(maxsize=TP.MAX_THREADS)
        self.futures = set()
        self.futures_lock = Lock()

        # Will keep track of tasks that are timing out here and kill them
        # eventually. This is needed given the task limit imposed by the above
        # queue.
        self.timeouts = dict()

        # We want to run futures which are never waited on otherwise. This
        # thread will do this.
        self.finisher_thread = Thread(target=self.finisher)
        self.finisher_thread.start()

        self.completed_tasks = 0
        self.timedout_tasks = 0
        self.failed_tasks = 0


    def _thread_starter(self, func, args, kwargs) -> Future:
        #present_stack = stack()

        future = self.thread_pool.submit(
            func, 
            *args,
            **kwargs
        )

        # print(future)

        return future

    """
    def finish_if_full(self):
        if self.futures.full():
            print("Task queue full. Finishing existing tasks.")
            self.finish()
    """

    """
    def runlater(self, func: Callable, *args, **kwargs) -> None:
        future = self._thread_starter(func, args, kwargs)

        # TODO bugfix
        self.futures.put(future)
    """

    def promise(self, func: Callable[..., T], *args, **kwargs) -> 'Future[T]':

        warnings.warn(
            "TP.promise will be deprecated. Use `TP.submit` or `TP.submit_robust` instead.",
            DeprecationWarning,
            stacklevel=2
        )

        return self.submit(func, *args, **kwargs)
    
    def submit(self, func: Callable[..., T], *args, **kwargs) -> 'Future[T]':
        """
        nonfull = False

        while not nonfull:
            with self.futures_lock:
                nonfull = len(self.futures) < TP.MAX_THREADS // 2
            if not nonfull:
                sleep(1)

        print(f"add {func.__name__}")
        """
        
        future = self._thread_starter(func, args, kwargs)

        with self.futures_lock:
            self.futures.add(future)

        return future

    def submit_robust(self, func: Callable[..., T], *args, **kwargs) -> 'Future[T]':
        # Submit an async task to run `func` wrapped with retry capabilities.

        def run(*args, **kwargs):
            retries: int = TP.ROBUST_RETRIES

            res = None
            future = self._thread_starter(func, args, kwargs)

            while res is None and retries > 0:
                try:
                    
                    res = future.result(timeout=TP.ROBUST_TIMEOUT)

                except TimeoutError as e:
                    logger.warning(f"Run of {func.__name__} in {threading.current_thread()} timed out. retries={retries}.")

                    #with self.futures_lock:
                    #    self.futures.remove(future)

                    future.cancel()
                    future = self._thread_starter(func, args, kwargs)

                    res = None
                    retries -= 1

                    if retries == 0:
                        raise e
                    
                # TODO: limit this to API/resource errors, don't include user errors that will always fail.
                except Exception as e:
                    logger.warning(f"Run of {func.__name__} in {threading.current_thread()} failed with {e}. retries={retries}.")

                    #with self.futures_lock:
                    #    self.futures.remove(future)

                    future.cancel()
                    future = self._thread_starter(func, args, kwargs)

                    res = None
                    retries -= 1

                    if retries == 0:
                        raise e
                    
            return res

        return self.submit(run, *args, **kwargs)

    def finisher(self):
        while True:
            if len(self.futures) == 0:
                sleep(1)

            try:
                with self.futures_lock:
                    futures = list(self.futures)

                #dones, not_dones = wait(futures, timeout=5)
                #print(f"done/not_done={len(dones)}/{len(not_dones)}")

                print(f"waiting for {len(futures)} futures")

                for f in as_completed(futures, timeout=1):
                    # print(f"remove {f}")

                    with self.futures_lock:
                        self.futures.remove(f)
    
                    try:
                        f.result()
                        self.completed_tasks += 1

                    except TimeoutError:
                        logger.warning(f"Run of {f} timed out.")
                        self.timedout_tasks += 1
                        
                    except Exception as e:
                        logger.warning(f"Run of {f} failed with {e}.")
                        self.failed_tasks += 1

            except TimeoutError as e:
                # print(e)
                pass

            #for f in self.thread_pool.
            #    self.finish()

    """
    def finish(self, timeout: Optional[float] = 5.0) -> int:
        # TODO bugfix
        # return

        logger.debug(f"Finishing {self.futures.qsize()} task(s).")

        timeouts = []

        # concurrent.futures.wait

        while not self.futures.empty():
            future = self.futures.get()
            try:
                future.result(timeout=timeout)

                self.completed_tasks += 1

                if future in self.timeouts:
                    del self.timeouts[future]

            except TimeoutError:
                if future in self.timeouts:
                    self.timeouts[future] += 1
                else:
                    self.timeouts[future] = 1

                if self.timeouts[future] > 3:
                    warnings.warn(f"Task for {future} timed out 3 times. Stopping it.", RuntimeWarning, stacklevel=3)

                    del self.timeouts[future]
                    future.cancel()

                    self.timedout_tasks += 1

                else:
                    timeouts.append(future)

        for future in timeouts:
            self.futures.put(future)

        if len(timeouts) == 0:
            logger.debug("Done.")
        else:
            logger.debug("Some tasks timed out.")

        return len(timeouts)
    """
        
    """
    def _status(self) -> List[str]:
        import pandas as pd

        rows = []

        for p in self.thread_pool._pool:
            rows.append([p.is_alive(), str(p)])

        return pd.DataFrame(rows, columns=["alive", "thread"])
    """
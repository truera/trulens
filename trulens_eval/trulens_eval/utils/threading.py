"""
Multi-threading utilities.
"""


from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as fThreadPoolExecutor
from concurrent.futures import TimeoutError
from inspect import stack
import logging
import threading

from typing import Callable, TypeVar

from trulens_eval.utils.python import _future_target_wrapper, code_line
from trulens_eval.utils.python import SingletonPerName

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_NETWORK_TIMEOUT: float = 10.0 # seconds


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


class TP(SingletonPerName['TP']):  # "thread processing"

    # Store here stacks of calls to various thread starting methods so that we
    # can retrieve the trace of calls that caused a thread to start.

    MAX_THREADS = 128

    # How long to wait for any task before restarting it.
    DEBUG_TIMEOUT = 60.0 # TODO: adjust after dev

    def __init__(self):
        if hasattr(self, "thread_pool"):
            # Already initialized as per SingletonPerName mechanism.
            return

        # Run tasks started with this class using this pool.
        self.thread_pool = fThreadPoolExecutor(
            max_workers=TP.MAX_THREADS,
            thread_name_prefix="TP.submit"
        )

        # Keep a seperate pool for threads whose function is only to wait for
        # the tasks executed in the above pool. Keeping this seperate to prevent
        # the deadlock whereas the wait thread waits for a tasks which will
        # never be run because the thread pool is filled with wait threads.
        self.thread_pool_debug_tasks = ThreadPoolExecutor(
            max_workers=TP.MAX_THREADS,
            thread_name_prefix="TP._submit"
        )

        # Store the futures for the tasks started with this class here.
        # This enforces an upper bound on how many tasks can be queued at once.
        # self.futures = Queue(maxsize=TP.MAX_THREADS)
        # self.futures = set()
        # self.futures_lock = Lock()

        # Will keep track of tasks that are timing out here and kill them
        # eventually. This is needed given the task limit imposed by the above
        # queue.
        # self.timeouts = dict()

        # We want to run futures which are never waited on otherwise. This
        # thread will do this.
        # self.finisher_thread = Thread(target=self.finisher)
        # self.finisher_thread.start()

        self.completed_tasks = 0
        self.timedout_tasks = 0
        self.failed_tasks = 0

    def _run_with_timeout(self, func: Callable[..., T], *args, **kwargs) -> T:

        fut: 'Future[T]' = self.thread_pool.submit(func, *args, **kwargs)

        try:
            res: T = fut.result(timeout=TP.DEBUG_TIMEOUT)
            return res

        except TimeoutError as e:
            logger.error(
                f"Run of {func.__name__} in {threading.current_thread()} timed out after {TP.DEBUG_TIMEOUT} second(s).\n"
                f"{code_line(func)}"
            )

            raise e

        except Exception as e:
            logger.warning(
                f"Run of {func.__name__} in {threading.current_thread()} failed with: {e}"
            )
            raise e

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

    def submit(self, func: Callable[..., T], *args, **kwargs) -> 'Future[T]':
        return self._submit(func, *args, **kwargs)

    def _submit(self, func: Callable[..., T], *args, **kwargs) -> 'Future[T]':
        # Submit a concurrent tasks to run `func` with the given `args` and
        # `kwargs` but stop with error if it ever takes too long. This is only
        # meant for debugging purposes as we expect all concurrent tasks to have
        # their own retry/timeout capabilities.

        return self.thread_pool_debug_tasks.submit(
            self._run_with_timeout,
            func,
            *args,
            **kwargs
        )

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
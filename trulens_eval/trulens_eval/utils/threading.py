"""
Multi-threading utilities.
"""

from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as fThreadPoolExecutor
from concurrent.futures import TimeoutError
import contextvars
from inspect import stack
import logging
import threading
from typing import Callable, Optional, TypeVar

from trulens_eval.utils.python import _future_target_wrapper
from trulens_eval.utils.python import code_line
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import SingletonPerName

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_NETWORK_TIMEOUT: float = 10.0  # seconds


class ThreadPoolExecutor(fThreadPoolExecutor):
    """
    A ThreadPoolExecutor that keeps track of the stack prior to each thread's
    invocation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):
        present_stack = stack()
        present_context = contextvars.copy_context()
        return super().submit(
            _future_target_wrapper, present_stack, present_context, fn, *args,
            **kwargs
        )


# HACK002: Attempt other users of ThreadPoolExecutor to use our version.
import concurrent

concurrent.futures.ThreadPoolExecutor = ThreadPoolExecutor
concurrent.futures.thread.ThreadPoolExecutor = ThreadPoolExecutor

# HACK003: Hack to try to make langchain use our ThreadPoolExecutor as the above doesn't
# seem to do the trick.
try:
    import langchain_core
    langchain_core.runnables.config.ThreadPoolExecutor = ThreadPoolExecutor

    # Newer langchain_core uses ContextThreadPoolExecutor extending
    # ThreadPoolExecutor. We cannot reliable override
    # concurrent.futures.ThreadPoolExecutor before langchain_core is loaded so
    # lets just retrofit the base class afterwards:
    from langchain_core.runnables.config import ContextThreadPoolExecutor
    ContextThreadPoolExecutor.__bases__ = (ThreadPoolExecutor,)

    # TODO: ContextThreadPoolExecutor already maintains context so we no longer
    # need to do it for them but we still need to maintain call stack.

except Exception:
    pass


class TP(SingletonPerName['TP']):  # "thread processing"

    # Store here stacks of calls to various thread starting methods so that we
    # can retrieve the trace of calls that caused a thread to start.

    MAX_THREADS: int = 128

    # How long to wait for any task before restarting it.
    DEBUG_TIMEOUT: Optional[float] = 600.0  # [seconds], None to disable

    def __init__(self):
        if safe_hasattr(self, "thread_pool"):
            # Already initialized as per SingletonPerName mechanism.
            return

        # Run tasks started with this class using this pool.
        self.thread_pool = fThreadPoolExecutor(
            max_workers=TP.MAX_THREADS, thread_name_prefix="TP.submit"
        )

        # Keep a seperate pool for threads whose function is only to wait for
        # the tasks executed in the above pool. Keeping this seperate to prevent
        # the deadlock whereas the wait thread waits for a tasks which will
        # never be run because the thread pool is filled with wait threads.
        self.thread_pool_debug_tasks = ThreadPoolExecutor(
            max_workers=TP.MAX_THREADS,
            thread_name_prefix="TP.submit with debug timeout"
        )

        self.completed_tasks = 0
        self.timedout_tasks = 0
        self.failed_tasks = 0

    def _run_with_timeout(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> T:
        if timeout is None:
            timeout = TP.DEBUG_TIMEOUT

        fut: 'Future[T]' = self.thread_pool.submit(func, *args, **kwargs)

        try:
            res: T = fut.result(timeout=timeout)
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

    def submit(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> 'Future[T]':
        if timeout is None:
            timeout = TP.DEBUG_TIMEOUT

        # TODO(piotrm): need deadlock fixes here. If submit or _submit was called
        # earlier in the stack, do not use a threadpool to evaluate this task
        # and instead create a new thread for it. This prevents tasks in a
        # threadpool adding sub-tasks in the same threadpool which can lead to
        # deadlocks. Alternatively just raise an exception in those cases.

        return self._submit(func, *args, timeout=timeout, **kwargs)

    def _submit(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> 'Future[T]':
        if timeout is None:
            timeout = TP.DEBUG_TIMEOUT

        # Submit a concurrent tasks to run `func` with the given `args` and
        # `kwargs` but stop with error if it ever takes too long. This is only
        # meant for debugging purposes as we expect all concurrent tasks to have
        # their own retry/timeout capabilities.

        return self.thread_pool_debug_tasks.submit(
            self._run_with_timeout, func, *args, timeout=timeout, **kwargs
        )

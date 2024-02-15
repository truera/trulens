"""
# Threading Utilities
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor as fThreadPoolExecutor
from concurrent.futures import TimeoutError
import contextvars
from inspect import stack
import logging
import threading
from threading import Thread as fThread
from typing import Callable, Optional, TypeVar

from trulens_eval.utils.python import _future_target_wrapper
from trulens_eval.utils.python import code_line
from trulens_eval.utils.python import Future
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import SingletonPerName
from trulens_eval.utils.python import T

logger = logging.getLogger(__name__)

DEFAULT_NETWORK_TIMEOUT: float = 10.0  # seconds

A = TypeVar("A")


class Thread(fThread):
    """Thread that wraps target with stack/context tracking.
    
    App components that do not use this thread class might not be properly
    tracked."""

    def __init__(
        self,
        name=None,
        group=None,
        target=None,
        args=(),
        kwargs={},
        daemon=None
    ):
        present_stack = stack()
        present_context = contextvars.copy_context()

        fThread.__init__(
            self,
            name=name,
            group=group,
            target=_future_target_wrapper,
            args=(present_stack, present_context, target, *args),
            kwargs=kwargs,
            daemon=daemon
        )


# HACK007: Attempt to force other users of Thread to use our version instead.
import threading

threading.Thread = Thread


class ThreadPoolExecutor(fThreadPoolExecutor):
    """A ThreadPoolExecutor that keeps track of the stack prior to each thread's
    invocation.
    
    Apps that do not use this thread pool might not be properly tracked.
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


# HACK002: Attempt to make other users of ThreadPoolExecutor use our version
# instead. TODO: this may be redundant with the thread override above.
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


class TP(SingletonPerName):  # "thread processing"
    """Manager of thread pools.

    Singleton.
    """

    MAX_THREADS: int = 128
    """Maximum number of threads to run concurrently."""

    DEBUG_TIMEOUT: Optional[float] = 600.0  # [seconds], None to disable
    """How long to wait (seconds) for any task before restarting it."""

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
        func: Callable[[A], T],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> T:
        if timeout is None:
            timeout = TP.DEBUG_TIMEOUT

        fut: Future[T] = self.thread_pool.submit(func, *args, **kwargs)

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
        func: Callable[[A], T],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Future[T]:
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
        func: Callable[[A], T],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Future[T]:
        if timeout is None:
            timeout = TP.DEBUG_TIMEOUT

        # Submit a concurrent tasks to run `func` with the given `args` and
        # `kwargs` but stop with error if it ever takes too long. This is only
        # meant for debugging purposes as we expect all concurrent tasks to have
        # their own retry/timeout capabilities.

        return self.thread_pool_debug_tasks.submit(
            self._run_with_timeout, func, *args, timeout=timeout, **kwargs
        )

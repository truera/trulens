"""Threading Utilities."""

from __future__ import annotations

from concurrent import futures
from concurrent.futures import ThreadPoolExecutor as fThreadPoolExecutor
from concurrent.futures import TimeoutError
import contextvars
import inspect
import logging
import threading
from threading import Thread as fThread
from typing import Callable, Optional, TypeVar

from trulens.core._utils.pycompat import Future  # import style exception
from trulens.core.utils import python as python_utils

logger = logging.getLogger(__name__)

DEFAULT_NETWORK_TIMEOUT: float = 10.0  # seconds

A = TypeVar("A")
T = TypeVar("T")


class Thread(fThread):
    """Thread that wraps target with copy of context and stack.

    App components that do not use this thread class might not be properly
    tracked.

    Some libraries are doing something similar so this class may be less and
    less needed over time but is still needed at least for our own uses of
    threads.
    """

    def __init__(
        self,
        name=None,
        group=None,
        target=None,
        args=(),
        kwargs={},
        daemon=None,
    ):
        present_stack = python_utils.WeakWrapper(inspect.stack(0))
        present_context = contextvars.copy_context()

        fThread.__init__(
            self,
            name=name,
            group=group,
            target=present_context.run,
            args=(
                python_utils._future_target_wrapper,
                present_stack,
                target,
                *args,
            ),
            kwargs=kwargs,
            daemon=daemon,
        )


class ThreadPoolExecutor(fThreadPoolExecutor):
    """A ThreadPoolExecutor that keeps track of the stack prior to each thread's
    invocation.

    Apps that do not use this thread pool might not be properly tracked.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):
        present_stack = python_utils.WeakWrapper(inspect.stack(0))
        present_context = contextvars.copy_context()

        return super().submit(
            present_context.run,
            python_utils._future_target_wrapper,
            present_stack,
            fn,
            *args,
            **kwargs,
        )


# HACK007: Attempt to force other users of Thread to use our version instead.

threading.Thread = Thread

# HACK002: Attempt to make other users of ThreadPoolExecutor use our version
# instead. TODO: this may be redundant with the thread override above.

futures.ThreadPoolExecutor = ThreadPoolExecutor
futures.thread.ThreadPoolExecutor = ThreadPoolExecutor


class TP(metaclass=python_utils.SingletonPerNameMeta):  # "thread processing"
    """Manager of thread pools.

    Singleton.
    """

    MAX_THREADS: int = 128
    """Maximum number of threads to run concurrently."""

    DEBUG_TIMEOUT: Optional[float] = 600.0  # [seconds], None to disable
    """How long to wait (seconds) for any task before restarting it."""

    def __init__(self):
        # Run tasks started with this class using this pool.
        self.thread_pool = ThreadPoolExecutor(
            max_workers=TP.MAX_THREADS, thread_name_prefix="TP.submit"
        )

        # Keep a separate pool for threads whose function is only to wait for
        # the tasks executed in the above pool. Keeping this separate to prevent
        # the deadlock whereas the wait thread waits for a tasks which will
        # never be run because the thread pool is filled with wait threads.
        self.thread_pool_debug_tasks = ThreadPoolExecutor(
            max_workers=TP.MAX_THREADS,
            thread_name_prefix="TP.submit with debug timeout",
        )

        self.completed_tasks = 0
        self.timedout_tasks = 0
        self.failed_tasks = 0

    def _run_with_timeout(
        self,
        func: Callable[[A], T],
        *args,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> T:
        if timeout is None:
            timeout = TP.DEBUG_TIMEOUT

        fut: Future[T] = self.thread_pool.submit(func, *args, **kwargs)

        try:
            res: T = fut.result(timeout=timeout)
            return res

        except TimeoutError as e:
            logger.error(
                "Run of %s in %s timed out after %s second(s).\n%s",
                func.__name__,
                threading.current_thread(),
                TP.DEBUG_TIMEOUT,
                python_utils.code_line(func),
            )

            raise e

        except Exception as e:
            logger.warning(
                "Run of %s in %s failed with: %s",
                {func.__name__},
                threading.current_thread(),
                e,
            )

            raise e

    def submit(
        self,
        func: Callable[[A], T],
        *args,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Future[T]:
        """Submit a task to run.

        Args:
            func: Function to run.

            *args: Positional arguments to pass to the function.

            timeout: How long to wait for the task to complete before killing it.

            **kwargs: Keyword arguments to pass to the function.
        """

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
        **kwargs,
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

    def shutdown(self):
        """Shutdown the pools."""

        self.thread_pool.shutdown()
        self.thread_pool_debug_tasks.shutdown()

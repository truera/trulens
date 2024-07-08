"""
# Synchronization/Async Utilities

NOTE: we cannot name a module "async" as it is a python keyword.

## Synchronous vs. Asynchronous

Some functions in trulens_eval come with asynchronous versions. Those use "async
def" instead of "def" and typically start with the letter "a" in their name with
the rest matching their synchronous version.

Due to how python handles such functions and how they are executed, it is
relatively difficult to reshare code between the two versions. Asynchronous
functions are executed by an async loop (see
[EventLoop](https://docs.python.org/3/library/asyncio-eventloop.html)). Python
prevents any threads from having more than one running loop meaning one may not
be able to create one to run some async code if one has already been
created/running in the thread. The method `sync` here, used to convert an async
computation into a sync computation, needs to create a new thread. The impact of
this, whether overhead, or record info, is uncertain.

### What should be Sync/Async?

Try to have all internals be async but for users we may expose sync versions via
the `sync` method. If internals are async and don't need exposure, don't need to
provide a synced version.

"""

import asyncio
import inspect
import logging
from threading import current_thread
from typing import Awaitable, Callable, TypeVar, Union

import nest_asyncio

from trulens_eval.utils import python as mod_python_utils
from trulens_eval.utils import threading as mod_threading_utils

nest_asyncio.apply()

logger = logging.getLogger(__name__)

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")

MaybeAwaitable = Union[T, Awaitable[T]]
"""Awaitable or not.

May be checked with [isawaitable][inspect.isawaitable].
"""

CallableMaybeAwaitable = Union[Callable[[A], B], Callable[[A], Awaitable[B]]]
"""Function or coroutine function.

May be checked with
[is_really_coroutinefunction][trulens_eval.utils.python.is_really_coroutinefunction].
"""

CallableAwaitable = Callable[[A], Awaitable[B]]
"""Function that produces an awaitable / coroutine function."""

ThunkMaybeAwaitable = Union[mod_python_utils.Thunk[T],
                            mod_python_utils.Thunk[Awaitable[T]]]
"""Thunk or coroutine thunk. 

May be checked with
[is_really_coroutinefunction][trulens_eval.utils.python.is_really_coroutinefunction].
"""


async def desync(
    func: CallableMaybeAwaitable[A, T], *args, **kwargs
) -> T:  # effectively Awaitable[T]:
    """
    Run the given function asynchronously with the given args. If it is not
    asynchronous, will run in thread. Note: this has to be marked async since in
    some cases we cannot tell ahead of time that `func` is asynchronous so we
    may end up running it to produce a coroutine object which we then need to
    run asynchronously.
    """

    if mod_python_utils.is_really_coroutinefunction(func):
        return await func(*args, **kwargs)

    else:
        res = await asyncio.to_thread(func, *args, **kwargs)

        # HACK010: Might actually have been a coroutine after all.
        if inspect.iscoroutine(res):
            return await res
        else:
            return res


def sync(func: CallableMaybeAwaitable[A, T], *args, **kwargs) -> T:
    """
    Get result of calling function on the given args. If it is awaitable, will
    block until it is finished. Runs in a new thread in such cases.
    """

    if mod_python_utils.is_really_coroutinefunction(func):
        func: Callable[[A], Awaitable[T]]
        awaitable: Awaitable[T] = func(*args, **kwargs)

        # HACK010: Debugging here to make sure it is awaitable.
        assert inspect.isawaitable(awaitable)

        # Check if there is a running loop.
        try:
            loop = asyncio.get_running_loop()

        except Exception:
            # If not, we can create one here and run it until completion.
            loop = asyncio.new_event_loop()
            ret = loop.run_until_complete(awaitable)
            loop.close()
            return ret

        try:
            # If have nest_asyncio, can run in current thread.
            import nest_asyncio
            return loop.run_until_complete(awaitable)
        except:
            pass

        try:
            # If have nest_asyncio, can run in current thread.
            import nest_asyncio
            return loop.run_until_complete(awaitable)
        except:
            pass

        # Otherwise we cannot run a new loop in this thread so we create a
        # new thread to run the awaitable until completion.

        def run_in_new_loop():
            th: mod_threading_utils.Thread = current_thread()
            # Attach return value and possibly exception to thread object so we
            # can retrieve from the starter of the thread.
            th.ret = None
            th.error = None
            try:
                loop = asyncio.new_event_loop()
                th.ret = loop.run_until_complete(awaitable)
                loop.close()

            except Exception as e:
                th.error = e

        thread = mod_threading_utils.Thread(target=run_in_new_loop)

        # Start thread and wait for it to finish.
        thread.start()
        thread.join()

        # Get the return or error, return the return or raise the error.
        if thread.error is not None:
            raise thread.error
        else:
            return thread.ret

    else:
        func: Callable[[A], T]
        # Not a coroutine function, so do not need to sync anything.
        # HACK010: TODO: What if the inspect fails here too? We do some checks
        # in desync but not here.

        return func(*args, **kwargs)

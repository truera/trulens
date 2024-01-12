"""
# Synchronization/Async Utilities

NOTE: we cannot name a module "async" as it is a python keyword.

## Synchronous vs. Asynchronous

Some functions in trulens_eval come with asynchronous versions. Those use "async
def" instead of "def" and typically start with the letter "a" in their name with
the rest matching their synchronous version. Example:

```python
    @staticmethod def track_all_costs( ...

    @staticmethod async def atrack_all_costs( ...
```

Due to how python handles such functions and how they are executed, it is
relatively difficult to reshare code between the two versions. Asynchronous
functions are executed by an async loop (see
[https://docs.python.org/3/library/asyncio-eventloop.html](Event Loop)). Python
prevents any threads from having more than one running loop meaning one may not
be able to create one to run some async code if one has already been
created/running in the thread. The method `sync` here, used to convert an async
computation into a sync computation, needs to create a new thread. The impact of
this, whether overhead, or record info, is uncertain.

"""

import asyncio
import inspect
import logging
from threading import current_thread
from typing import Awaitable, Callable, TypeVar, Union

from trulens_eval.utils.python import T
from trulens_eval.utils.python import Thunk
from trulens_eval.utils.threading import Thread

logger = logging.getLogger(__name__)

A = TypeVar("A")
B = TypeVar("B")

# Awaitable or not. May be checked with inspect.isawaitable . 
MaybeAwaitable = Union[T, Awaitable[T]]

# Function or coroutine function. May be checked with
# inspect.iscoroutinefunction .
CallableMaybeAwaitable = Union[
    Callable[[A], B],
    Callable[[A], Awaitable[B]]
]

# Thunk or coroutine thunk. May be checked with inspect.iscoroutinefunction .
ThunkMaybeAwaitable = Union[
    Thunk[T],
    Thunk[Awaitable[T]]
]

async def desync(thunk: ThunkMaybeAwaitable[T]) -> T: # effectively Awaitable[T]
    """
    Create an async of the given sync thunk.
    """
    # Note: the "async" in front of the "def" is enough for python to create a
    # coroutine from this method meaning it produces an awaitable if it gets
    # called.

    # Need to do something better here. This will cause our async methods to
    # block.
    result_or_awaitable = thunk()

    if inspect.isawaitable(result_or_awaitable):
        return await result_or_awaitable
    else:
        return result_or_awaitable

def sync(thunk: ThunkMaybeAwaitable[T]) -> T:
    """
    Get result of thunk. If it is awaitable, will block until it is finished.
    Runs in a new thread in such cases.
    """
    # TODO: don't create a new thread if not in an existing loop.

    result_or_awaitable: MaybeAwaitable[T] = thunk()

    if inspect.isawaitable(result_or_awaitable):
        result_or_awaitable: Awaitable[T]

        def run_in_new_loop():
            th: Thread = current_thread()
            th.ret = None
            th.error = None
            try:
                loop = asyncio.new_event_loop()
                th.ret = loop.run_until_complete(result_or_awaitable)
            except Exception as e:
                th.error = e

        thread = Thread(
            target=run_in_new_loop
        )

        thread.start()
        thread.join()
        
        if thread.error is not None:
            raise thread.error
        else:
            return thread.ret

    else:
        result_or_awaitable: T

        return result_or_awaitable
"""
Utilities related to core python functionalities.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pprint import PrettyPrinter
from queue import Queue
from typing import Any, Callable, Dict, Hashable, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")
Thunk = Callable[[], T]


# Python call stack utilities

# Attribute name for storing a callstack in asyncio tasks.
STACK = "__tru_stack"

def caller_frame(offset=0) -> 'frame':
    """
    Get the caller's (of this function) frame. See
    https://docs.python.org/3/reference/datamodel.html#frame-objects .
    """

    return inspect.stack()[offset + 1].frame


def task_factory_with_stack(loop, coro, *args, **kwargs) -> Sequence['frame']:
    """
    A task factory that annotates created tasks with stacks of their parents.
    """

    parent_task = asyncio.current_task(loop=loop)
    task = asyncio.tasks.Task(coro=coro, loop=loop, *args, **kwargs)

    stack = [fi.frame for fi in inspect.stack()[2:]]

    if parent_task is not None:
        stack = merge_stacks(stack, parent_task.get_stack()[::-1])
        # skipping create_task and task_factory

    setattr(task, STACK, stack)

    return task


# Instrument new_event_loop to set the above task_factory upon creation:
original_new_event_loop = asyncio.events.new_event_loop


def _new_event_loop():
    loop = original_new_event_loop()
    loop.set_task_factory(task_factory_with_stack)
    return loop


asyncio.events.new_event_loop = _new_event_loop


def get_task_stack(task: asyncio.Task) -> Sequence['frame']:
    """
    Get the annotated stack (if available) on the given task.
    """
    if hasattr(task, STACK):
        return getattr(task, STACK)
    else:
        # get_stack order is reverse of inspect.stack:
        return task.get_stack()[::-1]


def merge_stacks(s1: Sequence['frame'],
                 s2: Sequence['frame']) -> Sequence['frame']:
    """
    Assuming `s1` is a subset of `s2`, combine the two stacks in presumed call
    order.
    """

    ret = []

    while len(s1) > 1:
        f = s1[0]
        s1 = s1[1:]

        ret.append(f)
        try:
            s2i = s2.index(f)
            for _ in range(s2i):
                ret.append(s2[0])
                s2 = s2[1:]

        except:
            pass

    return ret


def stack_with_tasks() -> Sequence['frame']:
    """
    Get the current stack (not including this function) with frames reaching
    across Tasks.
    """

    ret = [fi.frame for fi in inspect.stack()[1:]]  # skip stack_with_task_stack

    try:
        task_stack = get_task_stack(asyncio.current_task())

        return merge_stacks(ret, task_stack)

    except:
        return ret


def _future_target_wrapper(stack, func, *args, **kwargs):
    """
    Wrapper for a function that is started by threads. This is needed to
    record the call stack prior to thread creation as in python threads do
    not inherit the stack. Our instrumentation, however, relies on walking
    the stack and need to do this to the frames prior to thread starts.
    """

    # Keep this for looking up via get_first_local_in_call_stack .
    pre_start_stack = stack

    return func(*args, **kwargs)


def get_all_local_in_call_stack(
    key: str,
    func: Callable[[Callable], bool],
    offset: int = 1
) -> Iterator[Any]:
    """
    Get the value of the local variable named `key` in the stack at all of the
    frames executing a function which `func` recognizes (returns True on)
    starting from the top of the stack except `offset` top frames. Returns None
    if `func` does not recognize the correct function. Raises RuntimeError if a
    function is recognized but does not have `key` in its locals.

    This method works across threads as long as they are started using the TP
    class above.
    """

    logger.debug(f"Looking for local '{key}' in the stack.")

    frames = stack_with_tasks()[offset + 1:]  # + 1 to skip this method itself

    # Using queue for frames as additional frames may be added due to handling threads.
    q = Queue()
    for f in frames:
        q.put(f)

    while not q.empty():
        f = q.get()

        logger.debug(f"{f.f_code}")

        if id(f.f_code) == id(_future_target_wrapper.__code__):
            logger.debug(
                "Found thread starter frame. "
                "Will walk over frames prior to thread start."
            )
            locs = f.f_locals
            assert "pre_start_stack" in locs, "Pre thread start stack expected but not found."
            for fi in locs['pre_start_stack']:
                q.put(fi.frame)
            continue

        if func(f.f_code):
            logger.debug(f"looking via {func.__name__}; found {f}")
            locs = f.f_locals
            if key in locs:
                yield locs[key]
            else:
                raise KeyError(f"No local named '{key}' found in frame {f}.")

    return

def get_first_local_in_call_stack(
    key: str,
    func: Callable[[Callable], bool],
    offset: int = 1
) -> Optional[Any]:
    """
    Get the value of the local variable named `key` in the stack at the nearest
    frame executing a function which `func` recognizes (returns True on).
    Returns None if `func` does not recognize the correct function. Raises
    RuntimeError if a function is recognized but does not have `key` in its
    locals.

    This method works across threads as long as they are started using the TP
    class above.
    """

    try:
        return next(iter(get_all_local_in_call_stack(key, func, offset + 1)))
    except StopIteration:
        return None



# Class utilities


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

        k = cls.__name__, name

        if k not in cls.instances:
            logger.debug(
                f"*** Creating new {cls.__name__} singleton instance for name = {name} ***"
            )
            SingletonPerName.instances[k] = super().__new__(cls)

        return SingletonPerName.instances[k]



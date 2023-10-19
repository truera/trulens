"""
Utilities related to core python functionalities.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pprint import PrettyPrinter
from queue import Queue
from typing import (
    Any, Callable, Dict, Generic, Hashable, Iterator, Optional, Sequence, Type,
    TypeVar, Union
)

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")
Thunk = Callable[[], T]

# Function utilities.


def code_line(func) -> Optional[str]:
    """
    Get a string representation of the location of the given function `func`.
    """
    if hasattr(func, "__code__"):
        code = func.__code__
        return f"{code.co_filename}:{code.co_firstlineno}"
    else:
        return None


def locals_except(*exceptions):
    """
    Get caller's locals except for the named exceptions.
    """

    locs = caller_frame(offset=1).f_locals  # 1 to skip this call

    return {k: v for k, v in locs.items() if k not in exceptions}


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

    logger.debug("Getting cross-Task stacks. Current stack:")
    for f in ret:
        logger.debug(f"\t{f}")

    try:
        task_stack = get_task_stack(asyncio.current_task())

        logger.debug(f"Merging in stack from {asyncio.current_task()}:")
        for s in task_stack:
            logger.debug(f"\t{s}")

        temp = merge_stacks(ret, task_stack)
        logger.debug(f"Complete stack:")
        for f in temp:
            logger.debug(f"\t{f}")

        return temp

    except:
        return ret


def _future_target_wrapper(stack, func, *args, **kwargs):
    """
    Wrapper for a function that is started by threads. This is needed to
    record the call stack prior to thread creation as in python threads do
    not inherit the stack. Our instrumentation, however, relies on walking
    the stack and need to do this to the frames prior to thread starts.
    """

    # TODO: See if threading.stack_size([size]) can be used instead.

    # Keep this for looking up via get_first_local_in_call_stack .
    pre_start_stack = stack

    return func(*args, **kwargs)


def get_all_local_in_call_stack(
    key: str,
    func: Callable[[Callable], bool],
    offset: Optional[int] = 1,
    skip: Optional[Any] = None  # really frame
) -> Iterator[Any]:
    """
    Get the value of the local variable named `key` in the stack at all of the
    frames executing a function which `func` recognizes (returns True on)
    starting from the top of the stack except `offset` top frames. If `skip`
    frame is provided, it is skipped as well. Returns None if `func` does not
    recognize the correct function. Raises RuntimeError if a function is
    recognized but does not have `key` in its locals.

    This method works across threads as long as they are started using the TP
    class above.

    NOTE: `offset` is unreliable for skipping the intended frame when operating
    with async tasks. In those cases, the `skip` argument is more reliable.
    """

    logger.debug(f"Looking for local '{key}' in the stack.")

    if skip is not None:
        logger.debug(f"Will be skipping {skip}.")

    frames = stack_with_tasks()[1:]  # + 1 to skip this method itself
    # NOTE: skipping offset frames is done below since the full stack may need
    # to be reconstructed there.

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

        if offset is not None and offset > 0:
            offset -= 1
            continue

        if func(f.f_code):
            logger.debug(f"Looking via {func.__name__}; found {f}")
            if skip is not None and f == skip:
                logger.debug(f"Skipping.")
                continue

            locs = f.f_locals
            if key in locs:
                yield locs[key]
            else:
                raise KeyError(f"No local named '{key}' found in frame {f}.")

    return


def get_first_local_in_call_stack(
    key: str,
    func: Callable[[Callable], bool],
    offset: Optional[int] = 1,
    skip: Optional[Any] = None  # actually frame
) -> Optional[Any]:
    """
    Get the value of the local variable named `key` in the stack at the nearest
    frame executing a function which `func` recognizes (returns True on)
    starting from the top of the stack except `offset` top frames. If `skip`
    frame is provided, it is skipped as well. Returns None if `func` does not
    recognize the correct function. Raises RuntimeError if a function is
    recognized but does not have `key` in its locals.

    This method works across threads as long as they are started using the TP
    class above.

    NOTE: `offset` is unreliable for skipping the intended frame when operating
    with async tasks. In those cases, the `skip` argument is more reliable.
    """

    try:
        return next(
            iter(
                get_all_local_in_call_stack(
                    key, func, offset=offset + 1, skip=skip
                )
            )
        )
    except StopIteration:
        return None


# Class utilities

T = TypeVar("T")


class SingletonPerName(Generic[T]):
    """
    Class for creating singleton instances except there being one instance max,
    there is one max per different `name` argument. If `name` is never given,
    reverts to normal singleton behaviour.
    """

    # Hold singleton instances here.
    instances: Dict[Hashable, 'SingletonPerName'] = dict()

    def __new__(
        cls: Type[SingletonPerName[T]],
        *args,
        name: Optional[str] = None,
        **kwargs
    ) -> SingletonPerName[T]:
        """
        Create the singleton instance if it doesn't already exist and return it.
        """

        k = cls.__name__, name

        if k not in cls.instances:
            logger.debug(
                f"*** Creating new {cls.__name__} singleton instance for name = {name} ***"
            )
            SingletonPerName.instances[k] = super().__new__(cls)

        obj: cls = SingletonPerName.instances[k]

        return obj

"""
Utilities related to core python functionalities.
"""

from __future__ import annotations

import asyncio
from concurrent import futures
import inspect
import logging
from pprint import PrettyPrinter
import queue
import sys
from typing import (Any, Callable, Dict, Generic, Hashable, Iterator, Optional,
                    Sequence, Type, TypeVar)


if sys.version_info >= (3, 9):
    Future = futures.Future
    """Alias for [concurrent.futures.Future][].
    
    In python < 3.9, a sublcass of [concurrent.futures.Future][] with
    `Generic[A]` is used instead.
    """

    Queue = queue.Queue
    """Alias for [queue.Queue][] .
    
    In python < 3.9, a sublcass of [queue.Queue][] with
    `Generic[A]` is used instead.
    """

else:
    # Fake classes which can have type args. In python earlier than 3.9, the
    # classes imported above cannot have type args which is annoying for type
    # annotations. We use these fake ones instead.

    A = TypeVar("A")

    # HACK011
    class Future(Generic[A], futures.Future):
        """Alias for [concurrent.futures.Future][].
    
        In python < 3.9, a sublcass of [concurrent.futures.Future][] with
        `Generic[A]` is used instead.
        """

    # HACK012
    class Queue(Generic[A], queue.Queue):
        """Alias for [queue.Queue][] .
    
        In python < 3.9, a sublcass of [queue.Queue][] with
        `Generic[A]` is used instead.
        """

if sys.version_info >= (3, 10):
    import types
    NoneType = types.NoneType
    """Alias for [types.NoneType][] .
    
    In python < 3.10, it is defined as `type(None)` instead.
    """

else:
    NoneType = type(None)
    """Alias for [types.NoneType][] .
    
    In python < 3.10, it is defined as `type(None)` instead.
    """

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")

Thunk = Callable[[], T]
"""A function that takes no arguments."""

# Reflection utilities.


def is_really_coroutinefunction(func) -> bool:
    """Determine whether the given function is a coroutine function.

    !!! Warning
     
        Inspect checkers for async functions do not work on openai clients,
        perhaps because they use `@typing.overload`. Because of that, we detect
        them by checking `__wrapped__` attribute instead. Note that the inspect
        docs suggest they should be able to handle wrapped functions but perhaps
        they handle different type of wrapping? See
        https://docs.python.org/3/library/inspect.html#inspect.iscoroutinefunction
        . Another place they do not work is the decorator langchain uses to mark
        deprecated functions.
    """

    if inspect.iscoroutinefunction(func):
        return True

    if hasattr(func, "__wrapped__") and inspect.iscoroutinefunction(
            func.__wrapped__):
        return True

    return False


def safe_signature(func_or_obj: Any):
    try:
        assert isinstance(
            func_or_obj, Callable
        ), f"Expected a Callable. Got {type(func_or_obj)} instead."

        return inspect.signature(func_or_obj)

    except Exception as e:
        if safe_hasattr(func_or_obj, "__call__"):
            # If given an obj that is callable (has __call__ defined), we want to
            # return signature of that call instead of letting inspect.signature
            # explore that object further. Doing so may produce exceptions due to
            # contents of those objects producing exceptions when attempting to
            # retrieve them.

            return inspect.signature(func_or_obj.__call__)

        else:
            raise e


def safe_hasattr(obj: Any, k: str) -> bool:
    try:
        v = inspect.getattr_static(obj, k)
    except AttributeError:
        return False

    is_prop = False
    try:
        # OpenAI version 1 classes may cause this isinstance test to raise an
        # exception.
        is_prop = isinstance(v, property)
    except Exception:
        return False

    if is_prop:
        try:
            v.fget(obj)
            return True
        except Exception as e:
            return False
    else:
        return True


# Function utilities.


def code_line(func) -> Optional[str]:
    """
    Get a string representation of the location of the given function `func`.
    """
    if safe_hasattr(func, "__code__"):
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
    if safe_hasattr(task, STACK):
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


def _future_target_wrapper(stack, context, func, *args, **kwargs):
    """
    Wrapper for a function that is started by threads. This is needed to
    record the call stack prior to thread creation as in python threads do
    not inherit the stack. Our instrumentation, however, relies on walking
    the stack and need to do this to the frames prior to thread starts.
    """

    # TODO: See if threading.stack_size([size]) can be used instead.

    # Keep this for looking up via get_first_local_in_call_stack .
    pre_start_stack = stack

    for var, value in context.items():
        var.set(value)

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

    frames = stack_with_tasks()[1:]  # + 1 to skip this method itself
    # NOTE: skipping offset frames is done below since the full stack may need
    # to be reconstructed there.

    # Using queue for frames as additional frames may be added due to handling threads.
    q = queue.Queue()
    for f in frames:
        q.put(f)

    while not q.empty():
        f = q.get()

        if id(f.f_code) == id(_future_target_wrapper.__code__):
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
        logger.debug("no frames found")
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
    _instances: Dict[Hashable, SingletonPerName] = dict()

    # Need some way to look up the name of the singleton instance. Cannot attach
    # a new attribute to instance since some metaclasses don't allow this (like
    # pydantic). We instead create a map from instance address to name.
    _id_to_name_map: Dict[int, Optional[str]] = dict()

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

        if k not in cls._instances:
            logger.debug(
                f"*** Creating new {cls.__name__} singleton instance for name = {name} ***"
            )
            # If exception happens here, the instance should not be added to
            # _instances.
            instance = super().__new__(cls)

            SingletonPerName._id_to_name_map[id(instance)] = name
            SingletonPerName._instances[k] = instance

        obj: cls = SingletonPerName._instances[k]

        return obj

    def delete_singleton(self):
        """
        Delete the singleton instance. Can be used for testing to create another
        singleton.
        """
        if id(self) in SingletonPerName._id_to_name_map:
            name = SingletonPerName._id_to_name_map[id(self)]
            del SingletonPerName._id_to_name_map[id(self)]
            del SingletonPerName._instances[self.__class__.__name__, name]
        else:
            logger.warning(f"Instance {self} not found in our records.")

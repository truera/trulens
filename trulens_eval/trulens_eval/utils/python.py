"""
Utilities related to core python functionalities.
"""

import inspect
import asyncio
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
Thunk = Callable[[], T]


def caller_frame(offset=0) -> 'frame':
    """
    Get the caller's (of this function) frame. See
    https://docs.python.org/3/reference/datamodel.html#frame-objects .
    """

    return inspect.stack()[offset + 1].frame


STACK = "__tru_stack"


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

    ret = [fi.frame for fi in inspect.stack()[1:]] # skip stack_with_task_stack

    try:
        task_stack = get_task_stack(asyncio.current_task())

        return merge_stacks(
            ret,  
            task_stack
        )
    
    except:
        return ret
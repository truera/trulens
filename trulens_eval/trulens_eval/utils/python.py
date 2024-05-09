"""
Utilities related to core python functionalities.
"""

from __future__ import annotations

import asyncio
from concurrent import futures
import dataclasses
import inspect
import logging
from pprint import PrettyPrinter
import queue
import sys
from types import ModuleType
import typing
from typing import (
    Any, Awaitable, Callable, Dict, Generator, Generic, Hashable, Iterator,
    List, Optional, Sequence, Type, TypeVar, Union
)

T = TypeVar("T")

Thunk = Callable[[], T]
"""A function that takes no arguments."""

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


class EmptyType(type):
    """A type that cannot be instantiated or subclassed."""

    def __new__(mcs, *args, **kwargs):
        raise ValueError("EmptyType cannot be instantiated.")

    def __instancecheck__(cls, __instance: Any) -> bool:
        return False

    def __subclasscheck__(cls, __subclass: Type) -> bool:
        return False


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

# Reflection utilities.


def class_name(obj: Union[Type, Any]) -> str:
    """Get the class name of the given object or instance."""

    if hasattr(obj, "__name__"):
        return obj.__name__

    if hasattr(obj, "__class__"):
        return obj.__class__.__name__

    return str(obj)


def module_name(obj: Union[ModuleType, Type, Any]) -> str:
    """Get the module name of the given module, class, or instance."""

    if isinstance(obj, ModuleType):
        return obj.__name__

    if hasattr(obj, "__module__"):
        return obj.__module__  # already a string name

    return "unknown module"


def callable_name(c: Callable):
    """Get the name of the given callable."""

    if not isinstance(c, Callable):
        raise ValueError(
            f"Expected a callable. Got {class_name(type(c))} instead."
        )

    if safe_hasattr(c, "__name__"):
        return c.__name__

    if safe_hasattr(c, "__call__"):
        return callable_name(c.__call__)

    return str(c)


def id_str(obj: Any) -> str:
    """Get the id of the given object as a string in hex."""

    return f"0x{id(obj):x}"


def is_really_coroutinefunction(func) -> bool:
    """Determine whether the given function is a coroutine function.

    Warning: 
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
    """Get the signature of the given function. 

    Sometimes signature fails for wrapped callables and in those cases we check
    for `__call__` attribute and use that instead.
    """
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
    """Check if the given object has the given attribute.
    
    Attempts to use static checks (see [inspect.getattr_static][]) to avoid any 
    side effects of attribute access (i.e. for properties).
    """
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
        except Exception:
            return False
    else:
        return True


def safe_issubclass(cls: Type, parent: Type) -> bool:
    """Check if the given class is a subclass of the given parent class."""

    origin = typing.get_origin(cls)
    if origin is None:
        return issubclass(cls, parent)

    return issubclass(origin, parent)


# Function utilities.


def code_line(func, show_source: bool = False) -> Optional[str]:
    """Get a string representation of the location of the given function
    `func`."""

    if isinstance(func, inspect.FrameInfo):
        ret = f"{func.filename}:{func.lineno}"
        if show_source:
            ret += "\n"
            for line in func.code_context:
                ret += "\t" + line

        return ret

    if inspect.isframe(func):
        code = func.f_code
        ret = f"{func.f_code.co_filename}:{func.f_code.co_firstlineno}"

    elif safe_hasattr(func, "__code__"):
        code = func.__code__
        ret = f"{code.co_filename}:{code.co_firstlineno}"

    else:
        return None

    if show_source:
        ret += "\n"
        for line in inspect.getsourcelines(func)[0]:
            ret += "\t" + str(line)

    return ret


def locals_except(*exceptions):
    """
    Get caller's locals except for the named exceptions.
    """

    locs = caller_frame(offset=1).f_locals  # 1 to skip this call

    return {k: v for k, v in locs.items() if k not in exceptions}


def for_all_methods(decorator, _except: Optional[List[str]] = None):
    """
    Applies decorator to all methods except classmethods, private methods and
    the ones specified with `_except`.
    """

    def decorate(cls):

        for attr_name, attr in cls.__dict__.items(
        ):  # does not include classmethods

            if not inspect.isfunction(attr):
                continue  # skips non-method attributes

            if attr_name.startswith("_"):
                continue  # skips private methods

            if _except is not None and attr_name in _except:
                continue

            logger.debug("Decorating %s", attr_name)
            setattr(cls, attr_name, decorator(attr))

        return cls

    return decorate


def run_before(callback: Callable):
    """
    Create decorator to run the callback before the function.
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            callback(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Python call stack utilities

# Attribute name for storing a callstack in asyncio tasks.
STACK = "__tru_stack"


def caller_frame(offset=0) -> 'frame':
    """
    Get the caller's (of this function) frame. See
    https://docs.python.org/3/reference/datamodel.html#frame-objects .
    """

    return inspect.stack()[offset + 1].frame


def caller_frameinfo(
    offset: int = 0,
    skip_module: Optional[str] = "trulens_eval"
) -> Optional[inspect.FrameInfo]:
    """
    Get the caller's (of this function) frameinfo. See
    https://docs.python.org/3/reference/datamodel.html#frame-objects .

    Args:
        offset: The number of frames to skip. Default is 0.
        
        skip_module: Skip frames from the given module. Default is "trulens_eval".
    """

    for finfo in inspect.stack()[offset + 1:]:
        if skip_module is None:
            return finfo
        if not finfo.frame.f_globals['__name__'].startswith(skip_module):
            return finfo

    return None


def task_factory_with_stack(loop, coro, *args, **kwargs) -> Sequence['frame']:
    """
    A task factory that annotates created tasks with stacks of their parents.
    
    All of such annotated stacks can be retrieved with
    [stack_with_tasks][trulens_eval.utils.python.stack_with_tasks] as one merged
    stack.
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
original_new_event_loop = asyncio.new_event_loop


def tru_new_event_loop():
    """
    Replacement for [new_event_loop][asyncio.new_event_loop] that sets
    the task factory to make tasks that copy the stack from their creators.
    """

    loop = original_new_event_loop()
    loop.set_task_factory(task_factory_with_stack)
    return loop


asyncio.new_event_loop = tru_new_event_loop


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
    """Find locals in call stack by name.
    
    Args:
        key: The name of the local variable to look for.
        
        func: Recognizer of the function to find in the call stack.
        
        offset: The number of top frames to skip.
        
        skip: A frame to skip as well.

    Note:
        `offset` is unreliable for skipping the intended frame when operating
        with async tasks. In those cases, the `skip` argument is more reliable.

    Returns:
        An iterator over the values of the local variable named `key` in the
            stack at all of the frames executing a function which `func` recognizes
            (returns True on) starting from the top of the stack except `offset` top
            frames.

            Returns None if `func` does not recognize any function in the stack.

    Raises:
        RuntimeError: Raised if a function is recognized but does not have `key`
            in its locals.

    This method works across threads as long as they are started using
    [TP][trulens_eval.utils.threading.TP].
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


# Wrapping utilities


class OpaqueWrapper(Generic[T]):
    """Wrap an object preventing all access.

    Any access except to
    [unwrap][trulens_eval.utils.python.OpaqueWrapper.unwrap] will result in an
    exception with the given message.

    Args:
        obj: The object to wrap.

        e: The exception to raise when an attribute is accessed.
    """

    def __init__(self, obj: T, e: Exception):
        self._obj = obj
        self._e = e

    def unwrap(self) -> T:
        """Get the wrapped object back."""
        return self._obj

    def __getattr__(self, name):
        raise self._e

    def __setattr__(self, name, value):
        if name in ["_obj", "_e"]:
            return super().__setattr__(name, value)
        raise self._e

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise self._e


def wrap_awaitable(
    awaitable: Awaitable[T],
    on_await: Optional[Callable[[], Any]] = None,
    on_done: Optional[Callable[[T], Any]] = None
) -> Awaitable[T]:
    """Wrap an awaitable in another awaitable that will call callbacks before
    and after the given awaitable finishes.

    Note that the resulting awaitable needs to be awaited for the callback to
    eventually trigger.

    Args:
        awaitable: The awaitable to wrap.

        on_await: The callback to call when the wrapper awaitable is awaited but
            before the wrapped awaitable is awaited.
        
        on_done: The callback to call with the result of the wrapped awaitable
            once it is ready.
    """

    async def wrapper(awaitable):
        if on_await is not None:
            on_await()

        val = await awaitable

        if on_done is not None:
            on_done(val)

        return val

    return wrapper(awaitable)


def wrap_generator(
    gen: Generator[T, None, None],
    on_iter: Optional[Callable[[], Any]] = None,
    on_next: Optional[Callable[[T], Any]] = None,
    on_done: Optional[Callable[[], Any]] = None
) -> Generator[T, None, None]:
    """Wrap a generator in another generator that will call callbacks at various
    points in the generation process.

    Args:
        gen: The generator to wrap.

        on_iter: The callback to call when the wrapper generator is created but
            before a first iteration is produced.

        on_next: The callback to call with the result of each iteration of the
            wrapped generator.

        on_done: The callback to call when the wrapped generator is exhausted.
    """

    def wrapper(gen):
        if on_iter is not None:
            on_iter()

        for val in gen:
            if on_next is not None:
                on_next(val)
            yield val

        if on_done is not None:
            on_done()

    return wrapper(gen)


# Class utilities

T = TypeVar("T")


@dataclasses.dataclass
class SingletonInfo(Generic[T]):
    """
    Information about a singleton instance.
    """

    val: T
    """The singleton instance."""

    name: str
    """The name of the singleton instance.
    
    This is used for the SingletonPerName mechanism to have a seperate singleton
    for each unique name (and class).
    """

    cls: Type[T]
    """The class of the singleton instance."""

    frame: Any
    """The frame where the singleton was created.
    
    This is used for showing "already created" warnings.
    """

    def __init__(self, name: str, val: Any):
        self.val = val
        self.cls = val.__class__
        self.name = name
        self.frameinfo = caller_frameinfo(offset=2)

    def warning(self):
        """Issue warning that this singleton already exists."""

        logger.warning(
            (
                "Singleton instance of type %s already created at:\n%s\n"
                "You can delete the singleton by calling `<instance>.delete_singleton()` or \n"
                f"""  ```python
  from trulens_eval.utils.python import SingletonPerName
  SingletonPerName.delete_singleton_by_name(name="{self.name}", cls={self.cls.__name__})
  ```
            """
            ), self.cls.__name__, code_line(self.frameinfo, show_source=True)
        )


class SingletonPerName(Generic[T]):
    """
    Class for creating singleton instances except there being one instance max,
    there is one max per different `name` argument. If `name` is never given,
    reverts to normal singleton behaviour.
    """

    # Hold singleton instances here.
    _instances: Dict[Hashable, SingletonInfo[SingletonPerName[T]]] = {}

    # Need some way to look up the name of the singleton instance. Cannot attach
    # a new attribute to instance since some metaclasses don't allow this (like
    # pydantic). We instead create a map from instance address to name.
    _id_to_name_map: Dict[int, Optional[str]] = {}

    def warning(self):
        """Issue warning that this singleton already exists."""

        name = SingletonPerName._id_to_name_map[id(self)]
        k = self.__class__.__name__, name
        if k in SingletonPerName._instances:
            SingletonPerName._instances[k].warning()
        else:
            raise RuntimeError(
                f"Instance of singleton type/name {k} does not exist."
            )

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
                "*** Creating new %s singleton instance for name = %s ***",
                cls.__name__, name
            )
            # If exception happens here, the instance should not be added to
            # _instances.
            instance = super().__new__(cls)

            SingletonPerName._id_to_name_map[id(instance)] = name
            info = SingletonInfo(name=name, val=instance)
            SingletonPerName._instances[k] = info
        else:
            info = SingletonPerName._instances[k]

        obj: cls = info.val

        return obj

    @staticmethod
    def delete_singleton_by_name(name: str, cls: Type[SingletonPerName] = None):
        """
        Delete the singleton instance with the given name.
        
        This can be used for testing to create another singleton.

        Args:
            name: The name of the singleton instance to delete.

            cls: The class of the singleton instance to delete. If not given, all
                instances with the given name are deleted.
        """
        for k, v in list(SingletonPerName._instances.items()):
            if k[1] == name:
                if cls is not None and v.cls != cls:
                    continue

                del SingletonPerName._instances[k]
                del SingletonPerName._id_to_name_map[id(v.val)]

    def delete_singleton(self):
        """
        Delete the singleton instance. Can be used for testing to create another
        singleton.
        """
        id_ = id(self)

        if id_ in SingletonPerName._id_to_name_map:
            name = SingletonPerName._id_to_name_map[id_]
            del SingletonPerName._id_to_name_map[id_]
            del SingletonPerName._instances[(self.__class__.__name__, name)]
        else:
            logger.warning("Instance %s not found in our records.", self)

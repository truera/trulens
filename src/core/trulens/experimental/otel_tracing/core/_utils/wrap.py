"""Utilities related to wrapping live python objects."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
import uuid

from trulens.core.utils import python as python_utils

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AwaitableCallbacks(Generic[T]):
    """Callbacks for wrapped awaitables.

    This class is intentially not an ABC so as to make default callbacks no-ops.
    """

    @staticmethod
    def on_awaitable_wrapped(
        awaitable: Awaitable[T], wrapper: Awaitable[T], **kwargs
    ) -> None:
        """Called immediately after the wrapper is produced."""

    def __init__(
        self,
        awaitable: Awaitable[T],
        wrapper: Awaitable[T],
        **kwargs: Dict[str, Any],
    ):
        """Called/constructed when wrapper awaitable is awaited but before the wrapped awaitable is awaited."""

        # Subclasses can use kwargs.

        self.awaitable: Awaitable[T] = awaitable
        self.wrapper: Awaitable[T] = wrapper

        self.result: Optional[T] = None
        self.error: Optional[Exception] = None

    def on_awaitable_end(self):
        """Called after the last callback is called."""

    def on_awaitable_result(self, result: T) -> T:
        """Called with the result of the wrapped awaitable once it is ready.

        !!! Important
            This should return the result or some wrapper of the result.
        """
        self.result = result
        return result

    def on_awaitable_exception(self, error: Exception) -> Exception:
        """Called if awaiting for the wrapped awaitable raised an exception."""
        self.error = error
        return error


def wrap_awaitable(
    awaitable: Awaitable[T],
    callback_class: Type[AwaitableCallbacks] = AwaitableCallbacks,
    **kwargs: Dict[str, Any],
) -> Awaitable[T]:
    """Wrap an awaitable in another awaitable that will call callbacks before
    and after the given awaitable finishes.
    Note that the resulting awaitable needs to be awaited for all but the
    `on_now` callback to trigger.
    Args:
        awaitable: The awaitable to wrap.
        callback_class: The class that provides callbacks.
        **kwargs: All other arguments are passed in as they are to all callbacks.
    """

    init_args: Dict[str, Any] = {"awaitable": awaitable, **kwargs}

    async def wrapper(awaitable):
        cb: AwaitableCallbacks = callback_class(**init_args)

        try:
            result = await awaitable
            result = cb.on_awaitable_result(result=result)
            cb.on_awaitable_end()
            return result

        except Exception as e:
            e_wrapped = cb.on_awaitable_exception(error=e)
            cb.on_awaitable_end()

            if e == e_wrapped:
                raise e

            raise e_wrapped from e

    w = wrapper(awaitable)

    init_args["wrapper"] = w
    callback_class.on_awaitable_wrapped(**init_args)

    return w


class IterableCallbacks(Generic[T]):
    """Callbacks for wrapped iterables.

    This class is intentially not an ABC so as to make default callbacks no-ops.
    """

    @staticmethod
    def on_iterable_wrapped(
        itb: Iterable[T], wrapper: Iterable[T], **kwargs: Dict[str, Any]
    ):
        """Called immediately after wrapper is made."""

    def __init__(
        self, itb: Iterable[T], wrapper: Iterable[T], **kwargs: Dict[str, Any]
    ):
        """Called/constructed when the wrapped iterable is iterated (__iter__ is
        called) but before the wrapped iterable is iterated (__iter__ is
        called)."""

        # Subclasses can use kwargs.

        self.itb: Iterable[T] = itb
        self.wrapper: Iterable[T] = wrapper

        self.it: Optional[Iterator[T]] = None
        self.error: Optional[Exception] = None

    def on_iterable_end(self):
        """Called after the last callback is called."""

    def on_iterable_iter(
        self,
        it: Iterator[T],
    ) -> Iterator[T]:
        """Called after the wrapped iterator is produced (but before any items are produced).

        !!! Important
            Must produce the given iterator it or some wrapper.
        """
        self.it = it
        return it

    def on_iterable_iter_exception(self, error: Exception) -> Exception:
        """Called if an error is raised during __iter__."""

        self.error = error
        return error

    def on_iterable_next(self):
        """Called when the next item from the iterator is requested (but before it is produced)."""

    def on_iterable_next_result(self, val: T) -> T:
        """Called after the next item from the iterator is produced.

        !!! Important
            Must produce the given value val or some wrapper.
        """
        return val

    def on_iterable_next_exception(self, error: Exception):
        """Called if an error is raised during __next__."""
        self.error = error
        return error

    def on_iterable_stop(self):
        """Called after the iterator stops iterating (raises StopIteration)."""


def wrap_iterable(
    itb: Iterable[T],
    callback_class: Type[IterableCallbacks] = IterableCallbacks,
    **kwargs: Dict[str, Any],
) -> Iterator[T]:
    """Wrap an iterable to invoke various callbacks.

    See `IterableCallbacks` for callbacks.
    Args:
        itb: The iterator to wrap.
        **kwargs: All other arguments are passed in as they are to all callbacks.
    """

    init_args = {"itb": itb, **kwargs}

    def wrapper(itb):
        cb = callback_class(**init_args)

        try:
            it: Iterator[T] = iter(itb)
            it = cb.on_iterable_iter(it=it)

        except Exception as e:
            init_args["error"] = e
            e_wrapped = cb.on_iterable_iter_exception(**init_args)
            cb.on_iterable_end()

            if e == e_wrapped:
                raise e

            raise e_wrapped from e

        while True:
            cb.on_iterable_next()

            try:
                val = next(it)
                val = cb.on_iterable_next_result(val=val)
                yield val

            except StopIteration:
                cb.on_iterable_stop()
                cb.on_iterable_end()
                return

            except Exception as e:
                e_wrapped = cb.on_iterable_next_exception(error=e)
                cb.on_iterable_end()

                if e == e_wrapped:
                    raise e

                raise e_wrapped from e

    w = wrapper(itb)

    init_args["wrapper"] = w
    callback_class.on_iterable_wrapped(**init_args)

    return w


class CallableCallbacks(Generic[T]):
    """Callbacks for wrapped callables.

    This class is intentially not an ABC so as to make default callbacks no-ops.
    """

    @classmethod
    def on_callable_wrapped(
        cls, *, func: Callable, wrapper: Callable, **kwargs: Dict[str, Any]
    ):
        """Called immediately after the wrapper is produced.

        !!! NOTE
            There is only one wrapper produced for each wrapped function. If
            wrapping is done again for an already wrapped function, this
            callback will get called with the existing wrapper.
        """

    def __init__(
        self,
        *,
        call_id: uuid.UUID,
        func: Callable,
        sig: inspect.Signature,
        wrapper: Callable,
        call_args: Tuple[Any, ...],
        call_kwargs: Dict[str, Any],
    ):
        """Called/constructed when the wrapper function is called but before
        arguments are bound to the wrapped function's signature."""

        # Subclasses can use kwargs.

        self.call_id: uuid.UUID = call_id
        self.func: Callable = func
        self.sig: inspect.Signature = sig
        self.wrapper: Callable = wrapper
        self.call_args: Optional[Tuple[Any, ...]] = call_args
        self.call_kwargs: Optional[Dict[str, Any]] = call_kwargs

        self.bindings: Optional[inspect.BoundArguments] = None
        self.bind_error: Optional[Exception] = None

        self.error: Optional[Exception] = None

        self.ret: Optional[T] = None

    def on_callable_end(self):
        """Called after the last callback is called."""

    def on_callable_call(
        self,
        *,
        bindings: inspect.BoundArguments,
    ) -> inspect.BoundArguments:
        """Called before the execution of the wrapped method assuming its
        arguments can be bound.

        !!! Important
            This callback must return the bound arguments or wrappers of bound arguments.
        """
        # TODO: instrument lazy arguments
        self.bindings = bindings

        return bindings

    def on_callable_bind_error(
        self,
        *,
        error: TypeError,
    ):
        """Called if the wrapped method's arguments cannot be bound.

        Note that if this type of error occurs, none of the wrapped code
        actually executes. Also if this happens, the `on_callable_exception`
        handler is expected to be called as well after the wrapped func is
        called with the unbindable arguments. This is done to replicate the
        behavior of an unwrapped invocation.
        """
        self.bind_error = error

    def on_callable_return(self, *, ret: T) -> T:
        """Called after wrapped method returns without error.

        !!! Important
            This callback must return the return value or some wrapper of that value.
        Example:
            ```python
            if isinstance(ret, Awaitable):
                return wrap_awaitable(ret)
            elif isinstance(ret, Iterable):
                return wrap_generator(ret)
            else:
                return ret
            ```
        """
        self.ret = ret
        return ret

    def on_callable_exception(self, *, error: Exception) -> Exception:
        """Called after wrapped method raises exception."""
        self.error = error
        return error


CALLBACKS = "__tru_callbacks"


def wrap_callable(
    func: Callable,
    callback_class: Type[CallableCallbacks],
    **kwargs: Dict[str, Any],
) -> Callable:
    """Create a wrapper of the given function to emit various callbacks at
    different stages of its evaluation.
    !!! Warning
        Note that we only allow a single wrapper and a callback class to be used
        at once. Managing multiple receivers of the data collected must be
        managed outside of this wrapping utility.

    Args:
        func: The function to wrap.
        callback_class: The class that provides callbacks.
        **kwargs: All other arguments are passed in as they are to the
            callback_class constructor upon the call to the wrapped func.
    """

    if hasattr(func, "__wrapper__"):
        # For cases where multiple refs to func are captured before we we begin
        # wrapping it and each of the refs is wrapped. We don't want to re-wrap
        # as we would miss some callbacks. We thus store the wrapper in
        # func.__wrapper__ and check for it here. The rest of the call will
        # assume func is the wrapper and will add more callbacks instead of
        # creating a new wrapper.
        func = func.__wrapper__

    assert isinstance(func, Callable), f"Callable expected but got {func}."

    if python_utils.safe_hasattr(func, CALLBACKS):
        # If CALLBACKS is set, it was already a wrapper.

        # logger.warning("Function %s is already wrapped.", func)

        return func

    sig = inspect.signature(func)  # safe sig?
    our_callback_init_kwargs: Dict[str, Any] = {"func": func, "sig": sig}

    # If CALLBACKS is not set, create a wrapper and return it.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(
            "Calling instrumented method %s of type %s.", func, type(func)
        )

        call_id: uuid.UUID = uuid.uuid4()

        callback_class, other_callback_init_kwargs = getattr(wrapper, CALLBACKS)
        callback = callback_class(
            call_id=call_id,
            call_args=args,
            call_kwargs=kwargs,
            **our_callback_init_kwargs,
            **other_callback_init_kwargs,
        )

        try:
            bindings = sig.bind(*args, **kwargs)  # save and reuse sig
            callback.on_callable_call(bindings=bindings)

        except TypeError as e:
            callback.on_callable_bind_error(error=e)

            # NOTE: Not raising e here to make sure the exception raised by the
            # wrapper is identical to the one produced by calling the wrapped
            # function below:

        try:
            # Get the result of the wrapped function.
            ret = func(*args, **kwargs)

            ret = callback.on_callable_return(ret=ret)
            # Can override ret.

            callback.on_callable_end()

            return ret

        except Exception as e:
            # Exception on wrapped func call: ret = func(...)

            wrapped_e = callback.on_callable_exception(error=e)
            # Can override exception.

            callback.on_callable_end()

            raise wrapped_e  # intentionally not "from e"

        """
        # disabling Awaitable and other lazy value handling for now

        if isinstance(ret, Awaitable):
            common_args['awaitable'] = ret
            callback.on_async_start(**common_args)
            def response_callback(_, response):
                logger.debug("Handling endpoint %s.", callback.endpoint.name)
                callback.on_async_end(
                    **common_args,
                    result=response,
                )
                span.cost = callback.cost
            return wrap_awaitable(
                ret,
                on_result=response_callback
            )
        """

        # else ret is not Awaitable

    # Set our tracking attribute to tell whether something is already
    # instrumented onto both the sync and async version since either one
    # could be returned from this method.

    our_callback_init_kwargs["wrapper"] = wrapper

    setattr(wrapper, CALLBACKS, (callback_class, kwargs))
    # The second item in the tuple are the other kwargs that were passed into
    # wrap_callable which are provided to the callback_class constructor.

    wrapper.__func__ = func
    func.__wrapper__ = wrapper

    callback_class.on_callable_wrapped(**our_callback_init_kwargs, **kwargs)

    return wrapper

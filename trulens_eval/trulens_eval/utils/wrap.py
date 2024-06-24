"""
Utilities related to wrapping live python objects.
"""

from __future__ import annotations

import asyncio
from concurrent import futures
import dataclasses
import functools
import inspect
import logging
from pprint import PrettyPrinter
import queue
import sys
from types import ModuleType
import typing
from typing import (
    Any, Awaitable, Callable, Dict, Generator, Generic, Hashable, Iterable,
    Iterator, List, Optional, Protocol, Sequence, Tuple, Type, TypeVar, Union
)

from trulens_eval.utils.python import safe_hasattr

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
        self, awaitable: Awaitable[T], wrapper: Awaitable[T],
        **kwargs: Dict[str, Any]
    ):
        """Called/constructed immedietely before on_awaitable_await."""

    def on_awaitable_await(self, awaitable: Awaitable[T], **kwargs) -> None:
        """Called when wrapper awaitable is awaited but before the wrapped awaitable is awaited."""

    def on_awaitable_result(
        self, awaitable: Awaitable[T], result: T, **kwargs
    ) -> None:
        """Called with the result of the wrapped awaitable once it is ready."""

    def on_awaitable_exception(
        self, awaitable: Awaitable[T], error: Exception, **kwargs
    ) -> None:
        """Called if awaiting for the wrapped awaitable raised an exception."""


def wrap_awaitable(
    awaitable: Awaitable[T],
    callback_class: Type[AwaitableCallbacks] = AwaitableCallbacks,
    **kwargs: Dict[str, Any]
) -> Awaitable[T]:
    """Wrap an awaitable in another awaitable that will call callbacks before
    and after the given awaitable finishes.

    Note that the resulting awaitable needs to be awaited for all but the
    `on_now` callback to trigger.

    Args:
        awaitable: The awaitable to wrap.

        **kwargs: All other arguments are passed in as they are to all callbacks.
    """

    common_args: Dict[str, Any] = {'awaitable': awaitable, **kwargs}

    async def wrapper(awaitable):
        cb: AwaitableCallbacks = callback_class(**common_args)
        cb.on_awaitable_await(awaitable=awaitable, **kwargs)

        result: Optional[T] = None
        error: Optional[Exception] = None

        try:
            result = await awaitable
        except Exception as e:
            error = e

        if error is not None:
            cb.on_awaitable_exception(
                awaitable=awaitable, error=error, **kwargs
            )
            raise error

        cb.on_awaitable_result(awaitable=awaitable, result=result, **kwargs)

        return result

    w = wrapper(awaitable)

    common_args['wrapper'] = w
    callback_class.on_awaitable_wrapped(**common_args)

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
        """Called/constructed right before on_iterable_iter."""

    def on_iteratable_iter(
        self, itb: Iterable[T], wrapper: Iterable[T], **kwargs: Dict[str, Any]
    ):
        """Called when the wrapped iterable is iterated (__iter__ is called)."""

    def on_iterable_iter_result(
        self, itb: Iterable[T], wrapper: Iterable[T], it: Iterator[T],
        **kwargs: Dict[str, Any]
    ) -> Iterator[T]:
        """Called after the iterator is produced (but before any items are produced).
        
        !!! Important
            Must produce the given iterator it or some wrapper.
        """
        return it

    def on_iterable_iter_exception(
        self, itb: Iterable[T], wrapper: Iterable[T], error: Exception
    ):
        """Called if an error is raised during __iter__."""

    def on_iterable_next(
        self, itb: Iterable[T], wrapper: Iterable[T], it: Iterator[T]
    ):
        """Called when the next item from the iterator is requested (but before it is produced)."""

    def on_iterable_next_result(
        self, itb: Iterable[T], wrapper: Iterable[T], it: Iterator[T], val: T
    ) -> T:
        """Called after the next item from the iterator is produced.
        
        !!! Important
            Must produce the given value val or some wrapper.
        """
        return val

    def on_iterable_next_exception(
        self, itb: Iterable[T], wrapper: Iterable[T], it: Iterator[T],
        error: Exception
    ):
        """Called if an error is raised during __next__."""

    def on_iteration_end(
        self, itb: Iterable[T], wrapper: Iterable[T], it: Iterator[T]
    ):
        """Called after the iterator stops iterating (raises StopIteration)."""


def wrap_iterable(
    itb: Iterable[T],
    callback_class: Type[IterableCallbacks] = IterableCallbacks,
    **kwargs: Dict[str, Any]
) -> Iterator[T]:
    """Wrap an iterable to invoke various callbacks.
    
    See `IterableCallbacks` for callbacks.

    Args:
        itb: The iterator to wrap.

        **kwargs: All other arguments are passed in as they are to all callbacks.
    """

    common_args = {'itb': itb, **kwargs}

    def wrapper(itb):
        cb = callback_class(**common_args)

        try:
            cb.on_iteratable_iter(**common_args)
            it: Iterable[T] = iter(itb)
            common_args['it'] = it
            common_args['it'] = cb.on_iterable_iter_result(**common_args)

        except Exception as e:
            cb.on_iterable_iter_exception(**common_args, error=e)
            raise e

        assert isinstance(it, Iterator)

        while True:
            cb.on_iterable_next(**common_args)
            try:
                val = next(it)
                val = cb.on_iterable_next_result(**common_args, val=val)
                yield val
            except StopIteration:
                cb.on_iteration_end(**common_args)
                return
            except Exception as e:
                cb.on_iterable_next_exception(**common_args, error=e)
                raise e

    w = wrapper(itb)

    common_args['wrapper'] = w
    callback_class.on_iterable_wrapped(**common_args)

    return w


class CallableCallbacks(Generic[T]):
    """Callbacks for wrapped callables.
    
    This class is intentially not an ABC so as to make default callbacks no-ops.
    """

    @staticmethod
    def on_callable_wrapped(
        func: Callable, wrapper: Callable, **kwargs: Dict[str, Any]
    ):
        """Called immediately after the wrapper is produced."""

    def __init__(
        self, func: Callable, wrapper: Callable, **kwargs: Dict[str, Any]
    ):
        """Called/constructed right before on_callable_bind."""

    def on_callable_bind(
        self, func: Callable, wrapper: Callable, args: Tuple[str],
        kwargs: Dict[str, Any]
    ):
        """Called when the wrapper function is called but before the wrapped function is called.
        
        Arguments are not yet bound to func's args.
        """

    def on_callable_call(
        self, func: Callable, wrapper: Callable,
        bindings: inspect.BoundArguments
    ) -> inspect.BoundArguments:
        """Called before the execution of the wrapped method assuming its
        arguments can be bound.
        
        !!! Important
            This callback must return the bound arguments or wrappers of bound arguments.
        """
        # TODO: instrument lazy arguments
        return bindings

    def on_callable_bind_error(
        self, func: Callable, wrapper: Callable, error: Exception,
        args: Tuple[Any], kwargs: Dict[str, Any]
    ):
        """Called if the wrapped method's arguments cannot be bound.
        
        Note that if this type of error occurs, none of the wrapped code
        actually executes. Also if this happens, the `on_callable_exception`
        handler is expected to be called as well after the wrapped func is
        called with the unbindable arguments. This is done to replicate the
        behavior of an unwrapped invocation.
        """

    def on_callable_return(
        self, func: Callable, wrapper: Callable,
        bindings: inspect.BoundArguments, ret: T
    ) -> T:
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
        return ret

    def on_callable_exception(
        self, func: Callable, wrapper: Callable,
        bindings: Optional[inspect.BoundArguments], error: Exception
    ):
        """Called after wrapped method raises exception."""


CALLBACKS = "__tru_callbacks"


def wrap_callable(
    func: Callable, callback_class: Type[CallableCallbacks], **kwargs
) -> Callable:
    """Create a wrapper of the given function to emit various callbacks at
    different stages of its evaluation.
    
    Args:
        func: The function to wrap.

        **kwargs: All other arguments are passed in as they are to all callbacks.
    """

    if safe_hasattr(func, CALLBACKS):
        # Store the callback class that will handle calls to the wrapped
        # function in the CALLBACKS attribute. This will be used to invoke
        # appropriate callbacks when the wrapped function gets called.

        assert callback_class is getattr(
            func, CALLBACKS
        ), "only one callback class may be assigned for callable wrappers."

        return func

    common_args: Dict[str, Any] = {'func': func, **kwargs}

    # If INSTRUMENT is not set, create a wrapper method and return it.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(
            "Calling instrumented method %s of type %s.", func, type(func)
        )

        ret: Any = None  # return of wrapped call if it did not raise an exception

        callback_class = getattr(wrapper, CALLBACKS)
        callback = callback_class(**common_args)

        call_common_args: Dict[str, Any] = dict(
            common_args
        )  # copy common args for this call

        callback.on_callable_bind(**call_common_args, args=args, kwargs=kwargs)
        call_common_args['bindings'] = None

        try:
            call_common_args['bindings'] = inspect.signature(func).bind(
                *args, **kwargs
            )
            callback.on_callable_call(**call_common_args)

        except Exception as e:
            callback.on_callable_bind_error(
                **call_common_args, error=e, args=args, kwargs=kwargs
            )
            # raise e

            # NOTE: Not raising e here to make sure the exception raised by the
            # wrapper is identical to the one produced by calling the wrapped
            # function below:

        try:
            # Get the result of the wrapped function.
            ret = func(*args, **kwargs)

        except Exception as e:
            # Or the exception it raised.
            callback.on_callable_exception(**call_common_args, error=e)
            raise e

        ret = callback.on_callable_return(**call_common_args, ret=ret)
        return ret

        # The rest of the code invokes the appropriate callbacks and then
        # populates the content of span created above.
        # span.endpoint = callback.endpoint
        # span.cost = callback.cost
        """
        if error is None and common_args['bindings'] is None:
            # If bindings is None, the `.bind()` line above should have failed.
            # In that case we expect the `func(...)` line to also fail. We bail
            # without the rest of the logic in that case while issuing a stern
            # warning.
            logger.warning("Wrapped function executed but we could not bind its arguments.")
            span.cost = callback.cost
            return ret

        if error is not None:
            callback.on_exception(**common_args, error=error)
            # common_args['bindings'] may be None in case the `.bind()` line
            # failed. The exception we caught when executing `func()` will
            # be the normal exception raised by bad function arguments which
            # is what we record in the cost span and reraise.
            span.error = str(error)
            span.cost = callback.cost
            raise error

        """
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
    setattr(wrapper, CALLBACKS, callback_class)

    common_args['wrapper'] = wrapper
    callback_class.on_callable_wrapped(**common_args)

    return wrapper

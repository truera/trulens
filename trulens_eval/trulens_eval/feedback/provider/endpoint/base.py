from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import functools
import inspect
import logging
from pprint import PrettyPrinter
from time import sleep
from types import ModuleType
from typing import (Any, Awaitable, Callable, ClassVar, Dict, List, Optional,
                    Sequence, Tuple, Type, TypeVar)

from pydantic import Field
import requests

from trulens_eval import trace as mod_trace
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.trace import SpanCost
from trulens_eval.utils import asynchro as mod_asynchro_utils
from trulens_eval.utils import pace as mod_pace
from trulens_eval.utils.pyschema import safe_getattr
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.python import callable_name
from trulens_eval.utils.python import class_name
from trulens_eval.utils.python import get_first_local_in_call_stack
from trulens_eval.utils.python import is_really_coroutinefunction
from trulens_eval.utils.python import module_name
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import SingletonPerName
from trulens_eval.utils.python import Thunk
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import SerialModel
from trulens_eval.utils.threading import DEFAULT_NETWORK_TIMEOUT
from trulens_eval.utils.wrap import wrap_awaitable

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")

INSTRUMENT = "__tru_instrument"

DEFAULT_RPM = 60
"""Default requests per minute for endpoints."""


class EndpointCallback(SerialModel):
    """Callbacks to be invoked after various API requests and track various metrics
    like token usage.
    """

    endpoint: Endpoint = Field(exclude=True)
    """The endpoint owning this callback."""

    cost: mod_base_schema.Cost = Field(default_factory=mod_base_schema.Cost)
    """Costs tracked by this callback."""

    def handle(self, response: Any) -> None:
        """Called after each request."""
        self.cost.n_requests += 1

    def handle_chunk(self, response: Any) -> None:
        """Called after receiving a chunk from a request."""
        self.cost.n_stream_chunks += 1

    def handle_generation(self, response: Any) -> None:
        """Called after each completion request."""
        self.handle(response)

    def handle_generation_chunk(self, response: Any) -> None:
        """Called after receiving a chunk from a completion request."""
        self.handle_chunk(response)

    def handle_classification(self, response: Any) -> None:
        """Called after each classification response."""
        self.handle(response)


class Endpoint(WithClassInfo, SerialModel, SingletonPerName):
    """API usage, pacing, and utilities for API endpoints."""

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    @dataclass
    class EndpointSetup():
        """Class for storing supported endpoint information.

        See [track_all_costs][trulens_eval.feedback.provider.endpoint.base.Endpoint.track_all_costs]
        for usage.
        """
        arg_flag: str
        module_name: str
        class_name: str

    ENDPOINT_SETUPS: ClassVar[List[EndpointSetup]] = [
        EndpointSetup(
            arg_flag="with_openai",
            module_name="trulens_eval.feedback.provider.endpoint.openai",
            class_name="OpenAIEndpoint"
        ),
        EndpointSetup(
            arg_flag="with_hugs",
            module_name="trulens_eval.feedback.provider.endpoint.hugs",
            class_name="HuggingfaceEndpoint"
        ),
        EndpointSetup(
            arg_flag="with_litellm",
            module_name="trulens_eval.feedback.provider.endpoint.litellm",
            class_name="LiteLLMEndpoint"
        ),
        EndpointSetup(
            arg_flag="with_bedrock",
            module_name="trulens_eval.feedback.provider.endpoint.bedrock",
            class_name="BedrockEndpoint"
        ),
        EndpointSetup(
            arg_flag="with_cortex",
            module_name="trulens_eval.feedback.provider.endpoint.cortex",
            class_name="CortexEndpoint"
        )
    ]

    instrumented_methods: ClassVar[Dict[Any, List[Tuple[Callable, Callable, Type[Endpoint]]]]] \
        = defaultdict(list)
    """Mapping of classe/module-methods that have been instrumented for cost
     tracking along with the wrapper methods and the class that instrumented
     them.
     
     Key is the class or module owning the instrumented method. Tuple
     value has:

        - original function,

        - wrapped version,

        - endpoint that did the wrapping.
     """

    name: str
    """API/endpoint name."""

    rpm: float = DEFAULT_RPM
    """Requests per minute."""

    retries: int = 3
    """Retries (if performing requests using this class)."""

    post_headers: Dict[str, str] = Field(default_factory=dict, exclude=True)
    """Optional post headers for post requests if done by this class."""

    pace: mod_pace.Pace = Field(
        default_factory=lambda: mod_pace.
        Pace(marks_per_second=DEFAULT_RPM / 60.0, seconds_per_period=60.0),
        exclude=True
    )
    """Pacing instance to maintain a desired rpm."""

    callback_class: Type[EndpointCallback] = Field(exclude=True)
    """Callback class to use for usage tracking."""

    callback_name: str = Field(exclude=True)
    """Name of variable that stores the callback noted above."""

    def __new__(cls, *args, name: Optional[str] = None, **kwargs):
        name = name or cls.__name__
        return super().__new__(cls, *args, name=name, **kwargs)

    def __str__(self):
        # Have to override str/repr due to pydantic issue with recursive models.
        return f"Endpoint({self.name})"

    def __repr__(self):
        # Have to override str/repr due to pydantic issue with recursive models.
        return f"Endpoint({self.name})"

    def __init__(
        self,
        *args,
        name: str,
        rpm: Optional[float] = None,
        callback_class: Optional[Any] = None,
        **kwargs
    ):
        if safe_hasattr(self, "rpm"):
            # already initialized via the SingletonPerName mechanism
            return

        if callback_class is None:
            # Some old databases do not have this serialized so lets set it to
            # the parent of callbacks and hope it never gets used.
            callback_class = EndpointCallback

        if rpm is None:
            rpm = DEFAULT_RPM

        kwargs['name'] = name
        kwargs['callback_class'] = callback_class
        kwargs['callback_name'] = f"callback_{name}"
        kwargs['pace'] = mod_pace.Pace(
            seconds_per_period=60.0,  # 1 minute
            marks_per_second=rpm / 60.0
        )

        super().__init__(*args, **kwargs)

        logger.debug("Creating new endpoint singleton with name %s.", self.name)

        # Extending class should call _instrument_module on the appropriate
        # modules and methods names.

    def pace_me(self) -> float:
        """Block until we can make a request to this endpoint to keep pace with
        maximum rpm.
        
        Returns time in seconds since last call to this method returned.
        """

        return self.pace.mark()

    def post(
        self,
        url: str,
        payload: JSON,
        timeout: float = DEFAULT_NETWORK_TIMEOUT
    ) -> Any:
        self.pace_me()
        ret = requests.post(
            url, json=payload, timeout=timeout, headers=self.post_headers
        )

        j = ret.json()

        # Huggingface public api sometimes tells us that a model is loading and
        # how long to wait:
        if "estimated_time" in j:
            wait_time = j['estimated_time']
            logger.error("Waiting for %s (%s) second(s).", j, wait_time)
            sleep(wait_time + 2)
            return self.post(url, payload)

        elif isinstance(j, Dict) and "error" in j:
            error = j['error']
            logger.error("API error: %s.", j)

            if error == "overloaded":
                logger.error("Waiting for overloaded API before trying again.")
                sleep(10.0)
                return self.post(url, payload)
            else:
                raise RuntimeError(error)

        assert isinstance(
            j, Sequence
        ) and len(j) > 0, f"Post did not return a sequence: {j}"

        if len(j) == 1:
            return j[0]

        else:
            return j

    def run_in_pace(self, func: Callable[[A], B], *args, **kwargs) -> B:
        """
        Run the given `func` on the given `args` and `kwargs` at pace with the
        endpoint-specified rpm. Failures will be retried `self.retries` times.
        """

        retries = self.retries + 1
        retry_delay = 2.0

        errors = []

        while retries > 0:
            try:
                self.pace_me()
                ret = func(*args, **kwargs)
                return ret

            except Exception as e:
                retries -= 1
                logger.error(
                    "%s request failed %s=%s. Retries remaining=%s.", self.name,
                    type(e), e, retries
                )
                errors.append(e)
                if retries > 0:
                    sleep(retry_delay)
                    retry_delay *= 2

        raise RuntimeError(
            f"Endpoint {self.name} request failed {self.retries+1} time(s): \n\t"
            + ("\n\t".join(map(str, errors)))
        )

    def run_me(self, thunk: Thunk[T]) -> T:
        """
        DEPRECTED: Run the given thunk, returning itse output, on pace with the api.
        Retries request multiple times if self.retries > 0.

        DEPRECATED: Use `run_in_pace` instead.
        """

        raise NotImplementedError(
            "This method is deprecated. Use `run_in_pace` instead."
        )

    def _instrument_module(self, mod: ModuleType, method_name: str) -> None:
        if safe_hasattr(mod, method_name):
            logger.debug(
                "Instrumenting %s.%s for %s", module_name(mod), method_name,
                self.name
            )
            func = getattr(mod, method_name)
            w = self.wrap_function(func)

            setattr(mod, method_name, w)

            Endpoint.instrumented_methods[mod].append((func, w, type(self)))

    def _instrument_class(self, cls, method_name: str) -> None:
        """Instrument the named method in the given class for cost tracking.

        Args:
            cls: The class to which the named method belongs.

            method_name: The method by name.
        """

        if safe_hasattr(cls, method_name):
            logger.debug(
                "Instrumenting %s.%s for %s", class_name(cls), method_name,
                self.name
            )
            func = getattr(cls, method_name)
            w = self.wrap_function(func)

            setattr(cls, method_name, w)

            Endpoint.instrumented_methods[cls].append((func, w, type(self)))

    @classmethod
    def print_instrumented(cls):
        """
        Print out all of the methods that have been instrumented for cost
        tracking. This is organized by the classes/modules containing them.
        """

        for wrapped_thing, wrappers in cls.instrumented_methods.items():
            print(
                wrapped_thing if wrapped_thing != object else
                "unknown dynamically generated class(es)"
            )
            for original, _, endpoint in wrappers:
                print(
                    f"\t`{original.__name__}` instrumented "
                    f"by {endpoint} at 0x{id(endpoint):x}"
                )

    def _instrument_class_wrapper(
        self, cls, wrapper_method_name: str,
        wrapped_method_filter: Callable[[Callable], bool]
    ) -> None:
        """
        Instrument a method `wrapper_method_name` which produces a method so
        that the produced method gets instrumented. Only instruments the
        produced methods if they are matched by named `wrapped_method_filter`.
        """
        if safe_hasattr(cls, wrapper_method_name):
            logger.debug(
                "Instrumenting method creator %s.%s for %s", cls.__name__,
                wrapper_method_name, self.name
            )
            func = getattr(cls, wrapper_method_name)

            def metawrap(*args, **kwargs):

                produced_func = func(*args, **kwargs)

                if wrapped_method_filter(produced_func):

                    logger.debug(
                        "Instrumenting %s", callable_name(produced_func)
                    )

                    instrumented_produced_func = self.wrap_function(
                        produced_func
                    )
                    Endpoint.instrumented_methods[object].append(
                        (produced_func, instrumented_produced_func, type(self))
                    )
                    return instrumented_produced_func
                else:
                    return produced_func

            Endpoint.instrumented_methods[cls].append(
                (func, metawrap, type(self))
            )

            setattr(cls, wrapper_method_name, metawrap)

    def _instrument_module_members(self, mod: ModuleType, method_name: str):
        """Instrument all of the given named methods in all of the classes
        belonging to the given module.

        Args:
            mod: The module whose members we want to instrument.

            method_name: The method we want to instrument (for all members that
            contain it).
        """

        if not safe_hasattr(mod, INSTRUMENT):
            setattr(mod, INSTRUMENT, set())

        already_instrumented = safe_getattr(mod, INSTRUMENT)

        if method_name in already_instrumented:
            logger.debug(
                "module %s already instrumented for %s", mod, method_name
            )
            return

        for m in dir(mod):
            logger.debug(
                "instrumenting module %s member %s for method %s", mod, m,
                method_name
            )
            if safe_hasattr(mod, m):
                obj = safe_getattr(mod, m)
                if inspect.isclass(obj):
                    self._instrument_class(obj, method_name=method_name)

        already_instrumented.add(method_name)

    def wrap_function(self, func: Callable):
        """Create a wrapper of the given function to perform cost tracking.
        
        Args:
            func: The function to wrap.
        """

        if safe_hasattr(func, INSTRUMENT):
            # Store the callback class that will handle calls to the wrapped
            # function in the INSTRUMENT attribute. This will be used to invoke
            # appropriate callbacks when the wrapped function gets called.

            # If INSTRUMENT is set, we don't need to instrument the method again
            # but we require that only one endpoint's EndpointCallback class is
            # registered to handle the wrapped call so we check any subsequent
            # wrap attempts are with the same endpoint.

            callback_class = getattr(func, INSTRUMENT)

            if not isinstance(callback_class, Type):
                raise TypeError(
                    "Wrapped INSTRUMENT attribute is expected to be a callback class."
                )

            if self.callback_class is not callback_class:
                raise ValueError(
                    f"Attempting to set more than one EndpointCallback for function {func}."
                )

            # If our callback class is already in the list, dont bother
            # adding it again.

            logger.warning(
                "%s already instrumented for callbacks of type %s for endpoint %s",
                func.__name__, self.callback_class.__name__,
                self.__class__.__name__
            )

            return func

        # If INSTRUMENT is not set, create a wrapper method and return it.
        @functools.wraps(func)
        def tru_wrapper(*args, **kwargs):
            logger.debug(
                "Calling instrumented method %s of type %s, "
                "iscoroutinefunction=%s, "
                "isasyncgeneratorfunction=%s", func, type(func),
                is_really_coroutinefunction(func),
                inspect.isasyncgenfunction(func)
            )

            ret: Any = None  # return of wrapped call if it did not raise an exception
            error: Any = None  # exception raised by wrapped call if any

            callback_class = getattr(tru_wrapper, INSTRUMENT)
            callback = callback_class(endpoint=self)

            common_args = {'func': func}
            callback.on_call(**common_args, args=args, kwargs=kwargs)
            common_args['bindings'] = None

            try:
                common_args['bindings'] = inspect.signature(func).bind(
                    *args, **kwargs
                )

            except Exception as e:
                callback.on_bind_error(
                    **common_args, error=e, args=args, kwargs=kwargs
                )
                # Continue to try to execute func which should fail and produce
                # some exception with regards to not being able to bind.

            with mod_trace.get_tracer().cost() as span:
                try:
                    # Get the result of the wrapped function.
                    ret = func(*args, **kwargs)

                except Exception as e:
                    # Or the exception it raised.
                    error = e

            # The rest of the code invokes the appropriate callbacks and then
            # populates the content of span created above.

            span.endpoint = callback.endpoint
            span.cost = callback.cost

            if error is None and common_args['bindings'] is None:
                # If bindings is None, the `.bind()` line above should have failed.
                # In that case we expect the `func(...)` line to also fail. We bail
                # without the rest of the logic in that case while issuing a stern
                # warning.
                logger.warning(
                    "Wrapped function executed but we could not bind its arguments."
                )
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

            callback.on_return(**common_args, ret=ret)
            return ret

        # Set our tracking attribute to tell whether something is already
        # instrumented onto both the sync and async version since either one
        # could be returned from this method.
        setattr(tru_wrapper, INSTRUMENT, self.callback_class)

        logger.debug("Instrumenting %s for %s.", func.__name__, self.name)

        return tru_wrapper


EndpointCallback.model_rebuild()
Endpoint.model_rebuild()

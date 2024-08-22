from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import functools
import importlib
import inspect
import logging
from pprint import PrettyPrinter
from time import sleep
from types import ModuleType
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from pydantic import Field
import requests
from trulens.core import preview as mod_preview
from trulens.core import trace as mod_trace
from trulens.core.schema import base as base_schema
from trulens.core.utils import asynchro as mod_asynchro_utils
from trulens.core.utils import pace as mod_pace
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import threading as threading_utils
from trulens.core.utils import wrap as wrap_utils

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")

DEFAULT_RPM = 60
"""Default requests per minute for endpoints."""


class WrapperEndpointCallback(mod_trace.TracingCallbacks[T]):
    """EXPERIMENTAL: otel-tracing

    Extension to TracingCallbacks that tracks costs.
    """

    # overriding CallableCallbacks
    def __init__(self, endpoint: Endpoint, **kwargs):
        super().__init__(**kwargs, span_type=mod_trace.LiveSpanCallWithCost)

        self.endpoint: Endpoint = endpoint
        self.span.endpoint = endpoint

        self.cost: base_schema.Cost = self.span.cost
        self.cost.n_requests += 1

    # overriding CallableCallbacks
    def on_callable_return(self, ret: T, **kwargs) -> T:
        """Called after a request returns."""

        self.cost.n_responses += 1

        ret = super().on_callable_return(ret=ret, **kwargs)
        # Fills in some general attributes from kwargs before the next callback
        # is called.

        self.on_endpoint_response(response=ret)

        return ret

    # our optional
    def on_endpoint_response(self, response: Any) -> None:
        """Called after each non-error response."""

        logger.warning("No on_endpoint_response method defined for %s.", self)

    # our optional
    def on_endpoint_generation(self, response: Any) -> None:
        """Called after each completion request."""

        self.cost.n_successful_requests += 1
        self.cost.n_generations += 1

    # our optional
    def on_endpoint_generation_chunk(self, response: Any) -> None:
        """Called after receiving a chunk from a completion request."""

        self.cost.n_stream_chunks += 1

    # our optional
    def on_endpoint_classification(self, response: Any) -> None:
        """Called after each classification response."""

        self.cost.n_successful_requests += 1
        self.cost.n_classifications += 1


class EndpointCallback(serial_utils.SerialModel):
    """Callbacks to be invoked after various API requests and track various metrics
    like token usage.
    """

    # TODEP: remove after EXPERIMENTAL: otel-tracing

    endpoint: Endpoint = Field(exclude=True)
    """The endpoint owning this callback."""

    cost: base_schema.Cost = Field(default_factory=base_schema.Cost)
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


class Endpoint(
    pyschema_utils.WithClassInfo,
    serial_utils.SerialModel,
    python_utils.SingletonPerName,
):
    """API usage, pacing, and utilities for API endpoints."""

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    @dataclass
    class EndpointSetup:
        """Class for storing supported endpoint information.

        See [track_all_costs][trulens.core.feedback.endpoint.Endpoint.track_all_costs]
        for usage.
        """

        # TODEP: remove after EXPERIMENTAL: otel-tracing

        arg_flag: str
        module_name: str
        class_name: str

    ENDPOINT_SETUPS: ClassVar[List[EndpointSetup]] = [
        EndpointSetup(
            arg_flag="with_openai",
            module_name="trulens.providers.openai.endpoint",
            class_name="OpenAIEndpoint",
        ),
        EndpointSetup(
            arg_flag="with_hugs",
            module_name="trulens.providers.huggingface.endpoint",
            class_name="HuggingfaceEndpoint",
        ),
        EndpointSetup(
            arg_flag="with_litellm",
            module_name="trulens.providers.litellm.endpoint",
            class_name="LiteLLMEndpoint",
        ),
        EndpointSetup(
            arg_flag="with_bedrock",
            module_name="trulens.providers.bedrock.endpoint",
            class_name="BedrockEndpoint",
        ),
        EndpointSetup(
            arg_flag="with_cortex",
            module_name="trulens.providers.cortex.endpoint",
            class_name="CortexEndpoint",
        ),
        EndpointSetup(
            arg_flag="with_dummy",
            module_name="trulens.feedback.dummy.endpoint",
            class_name="DummyEndpoint",
        ),
    ]
    """List of supported endpoints for tracking costs."""
    # TODEP: remove after EXPERIMENTAL: otel-tracing

    instrumented_methods: ClassVar[
        Dict[Any, List[Tuple[Callable, Callable, Type[Endpoint]]]]
    ] = defaultdict(list)
    """Mapping of classes/module-methods that have been instrumented for cost
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
        default_factory=lambda: mod_pace.Pace(
            marks_per_second=DEFAULT_RPM / 60.0, seconds_per_period=60.0
        ),
        exclude=True,
    )
    """Pacing instance to maintain a desired rpm."""

    global_callback: EndpointCallback = Field(
        exclude=True
    )  # of type _callback_class
    """Track costs not run inside "track_cost" here.

    Also note that Endpoints are singletons (one for each unique name argument)
    hence this global callback will track all requests for the named api even if
    you try to create multiple endpoints (with the same name).
    """
    # TODEP: remove after EXPERIMENTAL: otel-tracing

    callback_class: Type[EndpointCallback] = Field(exclude=True)
    """Callback class to use for usage tracking."""
    # TODEP: remove after EXPERIMENTAL: otel-tracing

    wrapper_callback_class: Type[WrapperEndpointCallback] = Field(exclude=True)
    """EXPERIMENTAL: otel-tracing

    Callback class to use for usage tracking.
    """

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
        wrapper_callback_class: Type[WrapperEndpointCallback] = None,
        **kwargs,
    ):
        if python_utils.safe_hasattr(self, "rpm"):
            # already initialized via the python_utils.SingletonPerName mechanism
            return

        if callback_class is None:
            # Some old databases do not have this serialized so lets set it to
            # the parent of callbacks and hope it never gets used.
            callback_class = EndpointCallback

        if rpm is None:
            rpm = DEFAULT_RPM

        kwargs["name"] = name
        kwargs["callback_class"] = callback_class
        kwargs["wrapper_callback_class"] = wrapper_callback_class
        kwargs["global_callback"] = callback_class(endpoint=self)
        kwargs["callback_name"] = f"callback_{name}"

        kwargs["pace"] = mod_pace.Pace(
            seconds_per_period=60.0,  # 1 minute
            marks_per_second=rpm / 60.0,
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
        payload: serial_utils.JSON,
        timeout: float = threading_utils.DEFAULT_NETWORK_TIMEOUT,
    ) -> Any:
        self.pace_me()
        ret = requests.post(
            url, json=payload, timeout=timeout, headers=self.post_headers
        )

        j = ret.json()

        # Huggingface public api sometimes tells us that a model is loading and
        # how long to wait:
        if "estimated_time" in j:
            wait_time = j["estimated_time"]
            logger.error("Waiting for %s (%s) second(s).", j, wait_time)
            sleep(wait_time + 2)
            return self.post(url, payload)

        elif isinstance(j, Dict) and "error" in j:
            error = j["error"]
            logger.error("API error: %s.", j)

            if error == "overloaded":
                logger.error("Waiting for overloaded API before trying again.")
                sleep(10.0)
                return self.post(url, payload)
            else:
                raise RuntimeError(error)

        assert (
            isinstance(j, Sequence) and len(j) > 0
        ), f"Post did not return a sequence: {j}"

        if len(j) == 1:
            return j[0]

        else:
            return j

    def run_in_pace(self, func: Callable[[A], B], *args, **kwargs) -> B:
        """Run the given `func` on the given `args` and `kwargs` at pace with the
        endpoint-specified rpm.

        Failures will be retried `self.retries` times.
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
                    "%s request failed %s=%s. Retries remaining=%s.",
                    self.name,
                    type(e),
                    e,
                    retries,
                )
                errors.append(e)
                if retries > 0:
                    sleep(retry_delay)
                    retry_delay *= 2

        raise RuntimeError(
            f"Endpoint {self.name} request failed {self.retries + 1} time(s): \n\t"
            + ("\n\t".join(map(str, errors)))
        )

    def run_me(self, thunk: python_utils.Thunk[T]) -> T:
        """DEPRECATED: Use `run_in_pace` instead."""

        raise NotImplementedError(
            "This method is deprecated. Use `run_in_pace` instead."
        )

    def _instrument_module(self, mod: ModuleType, method_name: str) -> None:
        if python_utils.safe_hasattr(mod, method_name):
            logger.debug(
                "Instrumenting %s.%s for %s",
                python_utils.module_name(mod),
                method_name,
                self.name,
            )
            func = getattr(mod, method_name)
            w = self.wrap_function(func)

            setattr(mod, method_name, w)

            Endpoint.instrumented_methods[mod].append((func, w, type(self)))

    def _instrument_class(self, cls, method_name: str) -> None:
        if python_utils.safe_hasattr(cls, method_name):
            logger.debug(
                "Instrumenting %s.%s for %s",
                python_utils.class_name(cls),
                method_name,
                self.name,
            )
            func = getattr(cls, method_name)
            w = self.wrap_function(func)

            setattr(cls, method_name, w)

            Endpoint.instrumented_methods[cls].append((func, w, type(self)))

    @classmethod
    def print_instrumented(cls):
        """Print out all of the methods that have been instrumented for cost
        tracking

        This is organized by the classes/modules containing them.
        """

        for wrapped_thing, wrappers in cls.instrumented_methods.items():
            print(
                wrapped_thing
                if wrapped_thing is not object
                else "unknown dynamically generated class(es)"
            )
            for original, _, endpoint in wrappers:
                print(
                    f"\t`{original.__name__}` instrumented "
                    f"by {endpoint} at 0x{id(endpoint):x}"
                )

    def _instrument_class_wrapper(
        self,
        cls,
        wrapper_method_name: str,
        wrapped_method_filter: Callable[[Callable], bool],
    ) -> None:
        """Instrument a method `wrapper_method_name` which produces a method so
        that the produced method gets instrumented.

        Only instruments the produced methods if they are matched by named
        `wrapped_method_filter`.
        """
        if python_utils.safe_hasattr(cls, wrapper_method_name):
            logger.debug(
                "Instrumenting method creator %s.%s for %s",
                cls.__name__,
                wrapper_method_name,
                self.name,
            )
            func = getattr(cls, wrapper_method_name)

            def metawrap(*args, **kwargs):
                produced_func = func(*args, **kwargs)

                if wrapped_method_filter(produced_func):
                    logger.debug(
                        "Instrumenting %s",
                        python_utils.callable_name(produced_func),
                    )

                    instrumented_produced_func = self.wrap_function(
                        produced_func
                    )
                    Endpoint.instrumented_methods[object].append((
                        produced_func,
                        instrumented_produced_func,
                        type(self),
                    ))
                    return instrumented_produced_func
                else:
                    return produced_func

            Endpoint.instrumented_methods[cls].append((
                func,
                metawrap,
                type(self),
            ))

            setattr(cls, wrapper_method_name, metawrap)

    def _instrument_module_members(self, mod: ModuleType, method_name: str):
        if not python_utils.safe_hasattr(mod, mod_trace.INSTRUMENT):
            setattr(mod, mod_trace.INSTRUMENT, set())

        already_instrumented = python_utils.safe_getattr(
            mod, mod_trace.INSTRUMENT
        )

        if method_name in already_instrumented:
            logger.debug(
                "module %s already instrumented for %s", mod, method_name
            )
            return

        for m in dir(mod):
            logger.debug(
                "instrumenting module %s member %s for method %s",
                mod,
                m,
                method_name,
            )
            if python_utils.safe_hasattr(mod, m):
                obj = python_utils.safe_getattr(mod, m)
                self._instrument_class(obj, method_name=method_name)

        already_instrumented.add(method_name)

    @staticmethod
    def track_all_costs(
        __func: mod_asynchro_utils.CallableMaybeAwaitable[A, T],
        *args,
        with_openai: bool = True,
        with_hugs: bool = True,
        with_litellm: bool = True,
        with_bedrock: bool = True,
        with_cortex: bool = True,
        with_dummy: bool = True,
        **kwargs,
    ) -> Tuple[T, Sequence[EndpointCallback]]:
        """Track costs of all of the apis we can currently track, over the
        execution of thunk.
        """
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        endpoints = []

        for endpoint in Endpoint.ENDPOINT_SETUPS:
            if locals().get(endpoint.arg_flag):
                try:
                    mod = importlib.import_module(endpoint.module_name)
                    cls = python_utils.safe_getattr(mod, endpoint.class_name)
                except Exception:
                    # If endpoint uses optional packages, will get either module
                    # not found error, or we will have a dummy which will fail
                    # at getattr. Skip either way.
                    continue

                try:
                    e = cls()
                    endpoints.append(e)

                except Exception as e:
                    logger.debug(
                        "Could not initialize endpoint %s. "
                        "Possibly missing key(s). "
                        "trulens will not track costs/usage of this endpoint. %s",
                        cls.__name__,
                        e,
                    )

        return Endpoint._track_costs(
            __func, *args, with_endpoints=endpoints, **kwargs
        )

    @staticmethod
    def track_all_costs_tally(
        __func: mod_asynchro_utils.CallableMaybeAwaitable[A, T],
        *args,
        with_openai: bool = True,
        with_hugs: bool = True,
        with_litellm: bool = True,
        with_bedrock: bool = True,
        with_cortex: bool = True,
        with_dummy: bool = True,
        **kwargs,
    ) -> Tuple[T, base_schema.Cost]:
        """Track costs of all of the apis we can currently track, over the
        execution of thunk.
        """
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        result, cbs = Endpoint.track_all_costs(
            __func,
            *args,
            with_openai=with_openai,
            with_hugs=with_hugs,
            with_litellm=with_litellm,
            with_bedrock=with_bedrock,
            with_cortex=with_cortex,
            with_dummy=with_dummy,
            **kwargs,
        )

        if len(cbs) == 0:
            # Otherwise sum returns "0" below.
            costs = base_schema.Cost()
        else:
            costs = sum(cb.cost for cb in cbs)

        return result, costs

    @staticmethod
    def _track_costs(
        __func: mod_asynchro_utils.CallableMaybeAwaitable[A, T],
        *args,
        with_endpoints: Optional[List[Endpoint]] = None,
        **kwargs,
    ) -> Tuple[T, Sequence[EndpointCallback]]:
        """Root of all cost tracking methods. Runs the given `thunk`, tracking
        costs using each of the provided endpoints' callbacks.
        """
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        # Check to see if this call is within another _track_costs call:
        endpoints: Dict[
            Type[EndpointCallback], List[Tuple[Endpoint, EndpointCallback]]
        ] = python_utils.get_first_local_in_call_stack(
            key="endpoints", func=Endpoint.__find_tracker, offset=1
        )

        if endpoints is None:
            # If not, lets start a new collection of endpoints here along with
            # the callbacks for each. See type above.

            endpoints = {}

        else:
            # We copy the dict here so that the outer call to _track_costs will
            # have their own version unaffacted by our additions below. Once
            # this frame returns, the outer frame will have its own endpoints
            # again and any wrapped method will get that smaller set of
            # endpoints.

            # TODO: check if deep copy is needed given we are storing lists in
            # the values and don't want to affect the existing ones here.
            endpoints = dict(endpoints)

        # Collect any new endpoints requested of us.
        with_endpoints = with_endpoints or []

        # Keep track of the new callback objects we create here for returning
        # later.
        callbacks = []

        # Create the callbacks for the new requested endpoints only. Existing
        # endpoints from other frames will keep their callbacks.
        for endpoint in with_endpoints:
            callback_class = endpoint.callback_class
            callback = callback_class(endpoint=endpoint)

            if callback_class not in endpoints:
                endpoints[callback_class] = []

            # And add them to the endpoints dict. This will be retrieved from
            # locals of this frame later in the wrapped methods.
            endpoints[callback_class].append((endpoint, callback))

            callbacks.append(callback)

        # Call the function.
        result: T = __func(*args, **kwargs)

        # Return result and only the callbacks created here. Outer thunks might
        # return others.
        return result, callbacks

    def track_cost(
        self,
        __func: mod_asynchro_utils.CallableMaybeAwaitable[T],
        *args,
        **kwargs,
    ) -> Tuple[T, EndpointCallback]:
        """Tally only the usage performed within the execution of the given thunk.

        Returns the thunk's result alongside the EndpointCallback object that
        includes the usage information.
        """
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        result, callbacks = Endpoint._track_costs(
            __func, *args, with_endpoints=[self], **kwargs
        )

        return result, callbacks[0]

    @staticmethod
    def __find_tracker(f):
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        return id(f) == id(Endpoint._track_costs.__code__)

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[EndpointCallback],
    ) -> None:
        """This gets called with the results of every instrumented method.

        This should be implemented by each subclass.

        Args:
            func: the wrapped method.

            bindings: the inputs to the wrapped method.

            response: whatever the wrapped function returned.

            callback: the callback set up by
                `track_cost` if the wrapped method was called and returned within an
                 invocation of `track_cost`.
        """
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        raise NotImplementedError(
            "Subclasses of Endpoint must implement handle_wrapped_call."
        )

    def _otel_wrap_function(self, func: Callable):
        """EXPERIMENTAL: otel-tracing

        Create a wrapper of the given function to perform cost tracking.

        Args:
            func: The function to wrap.
        """

        return wrap_utils.wrap_callable(
            func=func,
            func_name=python_utils.callable_name(func),
            callback_class=self.wrapper_callback_class,
            endpoint=self,
        )

    def _record_wrap_function(self, func):
        """Create a wrapper of the given function to perform cost tracking."""
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        if python_utils.safe_hasattr(func, mod_trace.INSTRUMENT):
            # Store the types of callback classes that will handle calls to the
            # wrapped function in the mod_trace.INSTRUMENT attribute. This will be used to
            # invoke appropriate callbacks when the wrapped function gets
            # called.

            # If mod_trace.INSTRUMENT is set, we don't need to instrument the method again
            # but we may need to add the additional callback class to expected
            # handlers stored at the attribute.

            registered_callback_classes = getattr(func, mod_trace.INSTRUMENT)

            if self.callback_class in registered_callback_classes:
                # If our callback class is already in the list, dont bother
                # adding it again.

                logger.debug(
                    "%s already instrumented for callbacks of type %s",
                    func.__name__,
                    self.callback_class.__name__,
                )

                return func

            else:
                # Otherwise add our callback class but don't instrument again.

                registered_callback_classes += [self.callback_class]
                setattr(func, mod_trace.INSTRUMENT, registered_callback_classes)

                return func

        # If mod_trace.INSTRUMENT is not set, create a wrapper method and return it.
        @functools.wraps(func)
        def tru_wrapper(*args, **kwargs):
            logger.debug(
                "Calling instrumented method %s of type %s, "
                "iscoroutinefunction=%s, "
                "isasyncgeneratorfunction=%s",
                func,
                type(func),
                python_utils.is_really_coroutinefunction(func),
                inspect.isasyncgenfunction(func),
            )

            # Get the result of the wrapped function:

            response = func(*args, **kwargs)

            bindings = inspect.signature(func).bind(*args, **kwargs)

            # Get all of the callback classes suitable for handling this
            # call. Note that we stored this in the mod_trace.INSTRUMENT attribute of
            # the wrapper method.
            registered_callback_classes = getattr(
                tru_wrapper, mod_trace.INSTRUMENT
            )

            # Look up the endpoints that are expecting to be notified and the
            # callback tracking the tally. See Endpoint._track_costs for
            # definition.
            endpoints: Dict[
                Type[EndpointCallback],
                Sequence[Tuple[Endpoint, EndpointCallback]],
            ] = python_utils.get_first_local_in_call_stack(
                key="endpoints", func=self.__find_tracker, offset=0
            )

            # If wrapped method was not called from within _track_costs, we
            # will get None here and do nothing but return wrapped
            # function's response.
            if endpoints is None:
                logger.debug("No endpoints found.")
                return response

            def response_callback(response):
                for callback_class in registered_callback_classes:
                    logger.debug("Handling callback_class: %s.", callback_class)
                    if callback_class not in endpoints:
                        logger.warning(
                            "Callback class %s is registered for handling %s"
                            " but there are no endpoints waiting to receive the result.",
                            callback_class.__name__,
                            func.__name__,
                        )
                        continue

                    for endpoint, callback in endpoints[callback_class]:
                        logger.debug("Handling endpoint %s.", endpoint.name)
                        endpoint.handle_wrapped_call(
                            func=func,
                            bindings=bindings,
                            response=response,
                            callback=callback,
                        )

            if isinstance(response, Awaitable):
                return wrap_utils.old_wrap_awaitable(
                    response, on_done=response_callback
                )

            response_callback(response)
            return response

        # Set our tracking attribute to tell whether something is already
        # instrumented onto both the sync and async version since either one
        # could be returned from this method.
        setattr(tru_wrapper, mod_trace.INSTRUMENT, [self.callback_class])

        logger.debug("Instrumenting %s for %s.", func.__name__, self.name)

        return tru_wrapper

    wrap_function = mod_preview.preview_method(
        mod_preview.Feature.OTEL_TRACING,
        enabled=_otel_wrap_function,
        disabled=_record_wrap_function,
    )


EndpointCallback.model_rebuild()
Endpoint.model_rebuild()

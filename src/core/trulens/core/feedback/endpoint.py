from __future__ import annotations

from collections import defaultdict
import contextvars
from dataclasses import dataclass
import functools
import importlib
import inspect
import logging
from pprint import PrettyPrinter
import re
from time import sleep
from types import ModuleType
from typing import (
    Any,
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

import pydantic
from pydantic import Field
import requests
from trulens.core.schema import base as base_schema
from trulens.core.utils import asynchro as asynchro_utils
from trulens.core.utils import pace as pace_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import threading as threading_utils

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")

INSTRUMENT = "__tru_instrument"

DEFAULT_RPM = 60
"""Default requests per minute for endpoints."""

_NO_CONTEXT_WARNING = """
Cannot find TruLens context. See
https://www.trulens.org/component_guides/other/no_context_warning for more information.
"""

_RE_NO_RETRY = re.compile(
    "("
    + ("|".join(["authentication", "unauthorized", "expired", "quota"]))
    + ")",
    re.IGNORECASE,
)
"""Pattern matched against request exceptions to determine whether they should
be aborted right away instead of retried."""


class EndpointCallback(serial_utils.SerialModel):
    """
    Callbacks to be invoked after various API requests and track various metrics
    like token usage.
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

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

        self.cost.n_completion_requests += 1

    def handle_generation_chunk(self, response: Any) -> None:
        """Called after receiving a chunk from a completion request."""

        self.handle_chunk(response)

    def handle_classification(self, response: Any) -> None:
        """Called after each classification response."""
        self.handle(response)

        self.cost.n_classification_requests += 1

    def handle_embedding(self, response: Any) -> None:
        """Called after each embedding response."""
        self.handle(response)

        self.cost.n_embedding_requests += 1


class Endpoint(
    pyschema_utils.WithClassInfo,
    serial_utils.SerialModel,
    python_utils.InstanceRefMixin,
):
    """API usage, pacing, and utilities for API endpoints."""

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

    @dataclass
    class EndpointSetup:
        """Class for storing supported endpoint information.

        See [track_all_costs][trulens.core.feedback.endpoint.Endpoint.track_all_costs]
        for usage.
        """

        arg_flag: str
        module_name: str
        class_name: str

    # TODO: factor this out
    BASE_ENDPOINTS: ClassVar[Dict[str, Endpoint]] = {}
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

    pace: pace_utils.Pace = Field(
        default_factory=lambda: pace_utils.Pace(
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

    callback_class: Type[EndpointCallback] = Field(exclude=True)
    """Callback class to use for usage tracking."""

    callback_name: str = Field(exclude=True)
    """Name of variable that stores the callback noted above."""

    _context_endpoints: ClassVar[contextvars.ContextVar] = (
        contextvars.ContextVar("endpoints", default={})
    )
    _context_endpoints.set({})

    def __str__(self):
        # Have to override str/repr due to pydantic issue with recursive models.
        return f"Endpoint({self.name})"

    def __repr__(self):
        # Have to override str/repr due to pydantic issue with recursive models.
        return f"Endpoint({self.name})"

    def __init__(
        self,
        *args,
        name: Optional[str] = None,
        rpm: Optional[float] = None,
        callback_class: Optional[Any] = None,
        _register_instance: bool = True,
        **kwargs,
    ):
        python_utils.InstanceRefMixin.__init__(
            self, register_instance=_register_instance
        )

        if callback_class is None:
            # Some old databases do not have this serialized so lets set it to
            # the parent of callbacks and hope it never gets used.
            callback_class = EndpointCallback
            # raise ValueError(
            #    "Endpoint has to be extended by class that can set `callback_class`."
            # )

        if rpm is None:
            rpm = DEFAULT_RPM
        if name is None:
            name = self.__class__.__name__

        kwargs["name"] = name
        kwargs["callback_class"] = callback_class
        kwargs["global_callback"] = callback_class(endpoint=self)
        kwargs["callback_name"] = f"callback_{name}"
        kwargs["pace"] = pace_utils.Pace(
            seconds_per_period=60.0,  # 1 minute
            marks_per_second=rpm / 60.0,
        )

        super().__init__(*args, **kwargs)

        logger.debug("Creating new endpoint singleton with name %s.", self.name)

        # Extending class should call _instrument_module on the appropriate
        # modules and methods names.

    def pace_me(self) -> float:
        """
        Block until we can make a request to this endpoint to keep pace with
        maximum rpm. Returns time in seconds since last call to this method
        returned.
        """

        return self.pace.mark()

    async def apace_me(self) -> float:
        yield self.pace.amark()

    def _can_retry(self, e: Exception) -> bool:
        """Determine whether a request that raised the given exception can be
        retried.

        Things like authorization errors should not be retried.
        """

        if _RE_NO_RETRY.search(str(e)) is not None:
            return False

        return True

    def run_in_pace(self, func: Callable[[A], B], *args, **kwargs) -> B:
        """Run the given `func` on the given `args` and `kwargs` at pace with the
        endpoint-specified rpm. Failures will be retried `self.retries` times."""

        retries = self.retries + 1
        attempts = 0
        retry_delay = 2.0

        errors = []

        while retries > 0:
            try:
                self.pace_me()
                attempts += 1
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
                if not self._can_retry(e):
                    break

                if retries > 0:
                    sleep(retry_delay)
                    retry_delay *= 2

        raise RuntimeError(
            f"Endpoint {self.name} request failed {attempts} time(s): \n\t"
            + ("\n\t".join(map(str, errors)))
        )

    def run_me(self, thunk: python_utils.Thunk[T]) -> T:
        """
        DEPRECATED: Run the given thunk, returning itse output, on pace with the api.
        Retries request multiple times if self.retries > 0.

        DEPRECATED: Use `run_in_pace` instead.
        """

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
        """
        Print out all of the methods that have been instrumented for cost
        tracking. This is organized by the classes/modules containing them.
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
        """
        Instrument a method `wrapper_method_name` which produces a method so
        that the produced method gets instrumented. Only instruments the
        produced methods if they are matched by named `wrapped_method_filter`.
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
        if not python_utils.safe_hasattr(mod, INSTRUMENT):
            setattr(mod, INSTRUMENT, set())

        already_instrumented = python_utils.safe_getattr(mod, INSTRUMENT)

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
            obj = python_utils.safer_getattr(mod, m)
            if obj is not None and isinstance(obj, type):
                # Instrument only classes, not instances.
                self._instrument_class(obj, method_name=method_name)

        already_instrumented.add(method_name)

    @staticmethod
    def track_all_costs(
        __func: asynchro_utils.CallableMaybeAwaitable[A, T],
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
        execution of thunk."""

        endpoints = []

        for endpoint_setup in Endpoint.ENDPOINT_SETUPS:
            if locals().get(endpoint_setup.arg_flag):
                try:
                    mod = importlib.import_module(endpoint_setup.module_name)
                    cls: Type[Endpoint] = python_utils.safe_getattr(
                        mod, endpoint_setup.class_name
                    )
                except ImportError:
                    # If endpoint uses optional packages, will get either module
                    # not found error, or we will have a dummy which will fail
                    # at getattr. Skip either way.
                    continue
                except Exception as e:
                    logger.debug(
                        "Could not import tracking module %s. "
                        "trulens will not track costs/usage of this endpoint. %s",
                        endpoint_setup.module_name,
                        e,
                    )
                    continue

                try:
                    endpoint = next(iter(cls.get_instances()))
                except StopIteration:
                    endpoint = None

                try:
                    if endpoint is None:
                        if cls.__name__ not in Endpoint.BASE_ENDPOINTS:
                            logger.debug(
                                "Could not find an instance of %s. "
                                "trulens will create an endpoint for cost tracking.",
                                cls.__name__,
                            )
                            Endpoint.BASE_ENDPOINTS[cls.__name__] = cls(
                                _register_instance=False
                            )
                        endpoint = Endpoint.BASE_ENDPOINTS[cls.__name__]

                    endpoints.append(endpoint)
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
        __func: asynchro_utils.CallableMaybeAwaitable[A, T],
        *args,
        with_openai: bool = True,
        with_hugs: bool = True,
        with_litellm: bool = True,
        with_bedrock: bool = True,
        with_cortex: bool = True,
        with_dummy: bool = True,
        **kwargs,
    ) -> Tuple[T, python_utils.Thunk[base_schema.Cost]]:
        """Track costs of all of the apis we can currently track, over the
        execution of thunk.

        Returns:
            T: Result of evaluating the thunk.

            Thunk[Cost]: A thunk that returns the total cost of all
                callbacks that tracked costs. This is a thunk as the costs might
                change after this method returns in case of Awaitable results.
        """

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
            tally = lambda: base_schema.Cost()
        else:
            tally = lambda: sum(cb.cost for cb in cbs)

        return result, tally

    @staticmethod
    def _track_costs(
        __func: asynchro_utils.CallableMaybeAwaitable[A, T],
        *args,
        with_endpoints: Optional[List[Endpoint]] = None,
        **kwargs,
    ) -> Tuple[T, Sequence[EndpointCallback]]:
        """Root of all cost tracking methods.

        Runs the given `thunk`, tracking costs using each of the provided
        endpoints' callbacks.
        """

        if (
            with_endpoints is not None
            and len(with_endpoints) > 0
            and not Endpoint._have_context()
        ):
            # If we cannot access the context vars, we cannot track costs, the
            # last condition issues a warning with more info.
            return __func(*args, **kwargs), []

        # Check to see if this call is within another _track_costs call:
        endpoints = dict(Endpoint._context_endpoints.get())  # copy

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

            # And add them to the endpoints dict.
            endpoints[callback_class].append((endpoint, callback))

            callbacks.append(callback)

        # Push the endpoints into the contextvars for wrappers inside the
        # following call to retrieve.
        endpoints_token = Endpoint._context_endpoints.set(endpoints)  # noqa: F841

        try:
            # context_vars = contextvars.copy_context()
            context_vars = {
                Endpoint._context_endpoints: Endpoint._context_endpoints.get()
            }

            # Call the function.
            result: T = __func(*args, **kwargs)

            def rewrap(result):
                if python_utils.is_lazy(result):
                    return python_utils.wrap_lazy(
                        result,
                        wrap=rewrap,
                        context_vars=context_vars,
                    )

                return result

            result = rewrap(result)
        finally:
            # Pop the endpoints from the contextvars.
            # Optionally disable to debug context issues. See App._set_context_vars.
            Endpoint._context_endpoints.reset(endpoints_token)

        # Return result and only the callbacks created here. Outer thunks might
        # return others.
        return result, callbacks

    def track_cost(
        self,
        __func: asynchro_utils.CallableMaybeAwaitable[..., T],
        *args,
        **kwargs,
    ) -> Tuple[T, EndpointCallback]:
        """Tally only the usage performed within the execution of the given
        thunk.

        Returns the thunk's result alongside the EndpointCallback object that
        includes the usage information.
        """

        result, callbacks = Endpoint._track_costs(
            __func, *args, with_endpoints=[self], **kwargs
        )

        return result, callbacks[0]

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[EndpointCallback],
    ) -> Any:
        """This gets called with the results of every instrumented method.

        This should be implemented by each subclass. Importantly, it must return
        the response or some wrapping of the response.

        Args:
            func: the wrapped method.

            bindings: the inputs to the wrapped method.

            response: whatever the wrapped function returned.

            callback: the callback set up by
                `track_cost` if the wrapped method was called and returned
                within an
                 invocation of `track_cost`.
        """
        return response

    @staticmethod
    def _have_context() -> bool:
        """Determine whether we can access the context vars needed for cost tracking."""

        try:
            Endpoint._context_endpoints.get()

        except LookupError:
            logger.warning(_NO_CONTEXT_WARNING)
            return False

        return True

    def wrap_function(self, func):
        """Create a wrapper of the given function to perform cost tracking."""

        if python_utils.safe_hasattr(func, INSTRUMENT):
            # Store the types of callback classes that will handle calls to the
            # wrapped function in the INSTRUMENT attribute. This will be used to
            # invoke appropriate callbacks when the wrapped function gets
            # called.

            # If INSTRUMENT is set, we don't need to instrument the method again
            # but we may need to add the additional callback class to expected
            # handlers stored at the attribute.

            registered_callback_classes = getattr(func, INSTRUMENT)

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
                setattr(func, INSTRUMENT, registered_callback_classes)

                return func

        # If INSTRUMENT is not set, create a wrapper method and return it.
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

            if not Endpoint._have_context():
                return func(*args, **kwargs)

            endpoints = Endpoint._context_endpoints.get()

            # Context vars as they are now before we reset them. They are
            # captured in the closure of the below.
            context_vars = {
                Endpoint._context_endpoints: Endpoint._context_endpoints.get()
            }

            # Get the result of the wrapped function:
            # response = context_vars.run(func, *args, **kwargs)
            response = func(*args, **kwargs)

            # if len(Endpoint._context_endpoints.get()) == 0:
            #    raise ValueError("No endpoints.")

            bindings = inspect.signature(func).bind(*args, **kwargs)

            # Get all of the callback classes suitable for handling this
            # call. Note that we stored this in the INSTRUMENT attribute of
            # the wrapper method.
            registered_callback_classes = getattr(tru_wrapper, INSTRUMENT)

            # If wrapped method was not called from within _track_costs, we
            # will get None here and do nothing but return wrapped
            # function's response.
            if len(endpoints) == 0:
                logger.debug("No endpoints found.")
                return response

            def update_response(response):
                if python_utils.is_lazy(response):
                    return python_utils.wrap_lazy(
                        response,
                        wrap=update_response,
                        context_vars=context_vars,
                    )

                for callback_class in registered_callback_classes:
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

                        with python_utils.with_context(context_vars):
                            response_ = endpoint.handle_wrapped_call(
                                func=func,
                                bindings=bindings,
                                response=response,
                                callback=callback,
                            )
                            if response_ is not None:
                                # Handler is allowed to override the response in
                                # case it wants to wrap some generators or similar
                                # lazy structures.
                                response = response_

                return response

            # Problem here: OpenAI returns generators inside its own special
            # classes. These are thus handled in
            # OpenAIEndpoint.wrapped_call .

            return update_response(response)

        # Set our tracking attribute to tell whether something is already
        # instrumented onto both the sync and async version since either one
        # could be returned from this method.
        setattr(tru_wrapper, INSTRUMENT, [self.callback_class])

        logger.debug("Instrumenting %s for %s.", func.__name__, self.name)

        return tru_wrapper


class _WithPost(Endpoint):
    """Endpoint with post methods."""

    post_headers: Dict[str, str] = Field(default_factory=dict)

    def post(
        self,
        url: str,
        json: serial_utils.JSON,
        timeout: Optional[float] = threading_utils.DEFAULT_NETWORK_TIMEOUT,
    ) -> requests.Response:
        """Make an http post request.

        Subclasses can include additional logic to handle endpoint-specific
        responses.
        """

        self.pace_me()

        return requests.post(
            url, json=json, timeout=timeout, headers=self.post_headers
        )

    async def apost(
        self,
        url: str,
        json: serial_utils.JSON,
        timeout: Optional[float] = threading_utils.DEFAULT_NETWORK_TIMEOUT,
    ) -> requests.Response:
        """Make an http post request.

        Subclasses can include additional logic to handle endpoint-specific
        responses.
        """

        await self.apace_me()

        # TODO: use an asynchronous post method.
        return requests.post(
            url, json=json, timeout=timeout, headers=self.post_headers
        )

    def post_json_first(
        self,
        url: str,
        json: serial_utils.JSON,
        timeout: float = threading_utils.DEFAULT_NETWORK_TIMEOUT,
    ) -> Dict:
        """Wraps `post` with json()[0]."""

        jdata = self.post(url=url, json=json, timeout=timeout).json()
        if len(jdata) == 0:
            raise ValueError("Empty response from post.")

        if len(jdata) > 1:
            logger.warning(
                "Received more than one response from post. "
                "Returning only the first."
            )

        return jdata[0]

    async def apost_json_first(
        self,
        url: str,
        json: serial_utils.JSON,
        timeout: float = threading_utils.DEFAULT_NETWORK_TIMEOUT,
    ) -> Dict:
        """Wraps `apost` with json()[0]."""

        jdata = (await self.apost(url=url, json=json, timeout=timeout)).json()

        if len(jdata) == 0:
            raise ValueError("Empty response from post.")

        if len(jdata) > 1:
            logger.warning(
                "Received more than one response from apost. "
                "Returning only the first."
            )

        return jdata[0]


EndpointCallback.model_rebuild()
Endpoint.model_rebuild()

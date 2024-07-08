from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import functools
import inspect
import logging
from pprint import PrettyPrinter
import random
import sys
from time import sleep
from types import ModuleType
from typing import (
    Any, Awaitable, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple,
    Type, TypeVar
)

from pydantic import Field
import requests

from trulens_eval.schema import base as mod_base_schema
from trulens_eval.utils import asynchro as mod_asynchro_utils
from trulens_eval.utils import pace as mod_pace
from trulens_eval.utils.pyschema import safe_getattr
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.python import callable_name
from trulens_eval.utils.python import class_name
from trulens_eval.utils.python import get_first_local_in_call_stack
from trulens_eval.utils.python import is_really_coroutinefunction
from trulens_eval.utils.python import locals_except
from trulens_eval.utils.python import module_name
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import SingletonPerName
from trulens_eval.utils.python import Thunk
from trulens_eval.utils.python import wrap_awaitable
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import SerialModel
from trulens_eval.utils.threading import DEFAULT_NETWORK_TIMEOUT

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")

INSTRUMENT = "__tru_instrument"

DEFAULT_RPM = 60
"""Default requests per minute for endpoints."""


class EndpointCallback(SerialModel):
    """
    Callbacks to be invoked after various API requests and track various metrics
    like token usage.
    """

    endpoint: Endpoint = Field(exclude=True)
    """Thhe endpoint owning this callback."""

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
            # raise ValueError(
            #    "Endpoint has to be extended by class that can set `callback_class`."
            # )

        if rpm is None:
            rpm = DEFAULT_RPM

        kwargs['name'] = name
        kwargs['callback_class'] = callback_class
        kwargs['global_callback'] = callback_class(endpoint=self)
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
        """
        Block until we can make a request to this endpoint to keep pace with
        maximum rpm. Returns time in seconds since last call to this method
        returned.
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
        **kwargs
    ) -> Tuple[T, Sequence[EndpointCallback]]:
        """
        Track costs of all of the apis we can currently track, over the
        execution of thunk.
        """

        endpoints = []

        for endpoint in Endpoint.ENDPOINT_SETUPS:
            if locals().get(endpoint.arg_flag):
                try:
                    mod = __import__(
                        endpoint.module_name, fromlist=[endpoint.class_name]
                    )
                    cls = safe_getattr(mod, endpoint.class_name)
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
                        "trulens_eval will not track costs/usage of this endpoint. %s",
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
        **kwargs
    ) -> Tuple[T, mod_base_schema.Cost]:
        """
        Track costs of all of the apis we can currently track, over the
        execution of thunk.
        """

        result, cbs = Endpoint.track_all_costs(
            __func,
            *args,
            with_openai=with_openai,
            with_hugs=with_hugs,
            with_litellm=with_litellm,
            with_bedrock=with_bedrock,
            with_cortex=with_cortex,
            **kwargs
        )

        if len(cbs) == 0:
            # Otherwise sum returns "0" below.
            costs = mod_base_schema.Cost()
        else:
            costs = sum(cb.cost for cb in cbs)

        return result, costs

    @staticmethod
    def _track_costs(
        __func: mod_asynchro_utils.CallableMaybeAwaitable[A, T],
        *args,
        with_endpoints: Optional[List[Endpoint]] = None,
        **kwargs
    ) -> Tuple[T, Sequence[EndpointCallback]]:
        """
        Root of all cost tracking methods. Runs the given `thunk`, tracking
        costs using each of the provided endpoints' callbacks.
        """
        # Check to see if this call is within another _track_costs call:
        endpoints: Dict[Type[EndpointCallback], List[Tuple[Endpoint, EndpointCallback]]] = \
            get_first_local_in_call_stack(
                key="endpoints",
                func=Endpoint.__find_tracker,
                offset=1
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
        self, __func: mod_asynchro_utils.CallableMaybeAwaitable[T], *args,
        **kwargs
    ) -> Tuple[T, EndpointCallback]:
        """
        Tally only the usage performed within the execution of the given thunk.
        Returns the thunk's result alongside the EndpointCallback object that
        includes the usage information.
        """

        result, callbacks = Endpoint._track_costs(
            __func, *args, with_endpoints=[self], **kwargs
        )

        return result, callbacks[0]

    @staticmethod
    def __find_tracker(f):
        return id(f) == id(Endpoint._track_costs.__code__)

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:
        """
        This gets called with the results of every instrumented method. This
        should be implemented by each subclass.

        Args:
            func: the wrapped method.

            bindings: the inputs to the wrapped method.

            response: whatever the wrapped function returned.

            callback: the callback set up by
                `track_cost` if the wrapped method was called and returned within an
                 invocation of `track_cost`.
        """
        raise NotImplementedError(
            "Subclasses of Endpoint must implement handle_wrapped_call."
        )

    def wrap_function(self, func):
        """Create a wrapper of the given function to perform cost tracking."""

        if safe_hasattr(func, INSTRUMENT):
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
                    func.__name__, self.callback_class.__name__
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
                "isasyncgeneratorfunction=%s", func, type(func),
                is_really_coroutinefunction(func),
                inspect.isasyncgenfunction(func)
            )

            # Get the result of the wrapped function:

            response = func(*args, **kwargs)

            bindings = inspect.signature(func).bind(*args, **kwargs)

            # Get all of the callback classes suitable for handling this
            # call. Note that we stored this in the INSTRUMENT attribute of
            # the wrapper method.
            registered_callback_classes = getattr(tru_wrapper, INSTRUMENT)

            # Look up the endpoints that are expecting to be notified and the
            # callback tracking the tally. See Endpoint._track_costs for
            # definition.
            endpoints: Dict[Type[EndpointCallback], Sequence[Tuple[Endpoint, EndpointCallback]]] = \
                get_first_local_in_call_stack(
                    key="endpoints",
                    func=self.__find_tracker,
                    offset=0
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
                            callback_class.__name__, func.__name__
                        )
                        continue

                    for endpoint, callback in endpoints[callback_class]:
                        logger.debug("Handling endpoint %s.", endpoint.name)
                        endpoint.handle_wrapped_call(
                            func=func,
                            bindings=bindings,
                            response=response,
                            callback=callback
                        )

            if isinstance(response, Awaitable):
                return wrap_awaitable(response, on_done=response_callback)

            response_callback(response)
            return response

        # Set our tracking attribute to tell whether something is already
        # instrumented onto both the sync and async version since either one
        # could be returned from this method.
        setattr(tru_wrapper, INSTRUMENT, [self.callback_class])

        logger.debug("Instrumenting %s for %s.", func.__name__, self.name)

        return tru_wrapper


class DummyEndpoint(Endpoint):
    """Endpoint for testing purposes.

    Does not make any network calls and just pretends to.
    """

    loading_prob: float
    """How often to produce the "model loading" response that huggingface api
    sometimes produces."""

    loading_time: Callable[[], float] = \
        Field(exclude=True, default_factory=lambda: lambda: random.uniform(0.73, 3.7))
    """How much time to indicate as needed to load the model in the above response."""

    error_prob: float
    """How often to produce an error response."""

    freeze_prob: float
    """How often to freeze instead of producing a response."""

    overloaded_prob: float
    """# How often to produce the overloaded message that huggingface sometimes produces."""

    alloc: int
    """How much data in bytes to allocate when making requests."""

    delay: float = 0.0
    """How long to delay each request."""

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="dummyendpoint")

    def __init__(
        self,
        name: str = "dummyendpoint",
        error_prob: float = 1 / 100,
        freeze_prob: float = 1 / 100,
        overloaded_prob: float = 1 / 100,
        loading_prob: float = 1 / 100,
        alloc: int = 1024 * 1024,
        delay: float = 0.0,
        rpm: float = DEFAULT_RPM * 10,
        **kwargs
    ):
        if safe_hasattr(self, "callback_class"):
            # Already created with SingletonPerName mechanism
            return

        assert error_prob + freeze_prob + overloaded_prob + \
            loading_prob <= 1.0, "Probabilites should not exceed 1.0 ."
        assert rpm > 0
        assert alloc >= 0
        assert delay >= 0.0

        kwargs['name'] = name
        kwargs['callback_class'] = EndpointCallback

        super().__init__(
            **kwargs, **locals_except("self", "name", "kwargs", "__class__")
        )

        logger.info(
            "Using DummyEndpoint with %s",
            locals_except('self', 'name', 'kwargs', '__class__')
        )

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:
        """Dummy handler does nothing."""

    def post(
        self, url: str, payload: JSON, timeout: Optional[float] = None
    ) -> Any:
        """Pretend to make a classification request similar to huggingface API.

        Simulates overloaded, model loading, frozen, error as configured:

        ```python
        requests.post(
            url, json=payload, timeout=timeout, headers=self.post_headers
        )
        ```

        """
        if timeout is None:
            timeout = DEFAULT_NETWORK_TIMEOUT

        self.pace_me()

        # allocate some data to pretend we are doing hard work
        temporary = [0x42] * self.alloc

        from numpy import random as np_random

        if self.delay > 0.0:
            sleep(max(0.0, np_random.normal(self.delay, self.delay / 2)))

        r = random.random()
        j: Optional[JSON] = None

        if r < self.freeze_prob:
            # Simulated freeze outcome.

            while True:
                sleep(timeout)
                raise TimeoutError()

        r -= self.freeze_prob

        if r < self.error_prob:
            # Simulated error outcome.

            raise RuntimeError("Simulated error happened.")
        r -= self.error_prob

        if r < self.loading_prob:
            # Simulated loading model outcome.

            j = {'estimated_time': self.loading_time()}
        r -= self.loading_prob

        if r < self.overloaded_prob:
            # Simulated overloaded outcome.

            j = {'error': "overloaded"}
        r -= self.overloaded_prob

        if j is None:
            # Otherwise a simulated success outcome with some constant results plus some randomness.

            j = [
                [
                    {
                        'label': 'LABEL_1',
                        'score': 0.6034979224205017 + random.random()
                    }, {
                        'label': 'LABEL_2',
                        'score': 0.2648237645626068 + random.random()
                    }, {
                        'label': 'LABEL_0',
                        'score': 0.13167837262153625 + random.random()
                    }
                ]
            ]

        # The rest is the same as in Endpoint:

        # Huggingface public api sometimes tells us that a model is loading and
        # how long to wait:
        if "estimated_time" in j:
            wait_time = j['estimated_time']
            logger.warning(
                "Waiting for %s (%s) second(s).",
                j,
                wait_time,
            )
            sleep(wait_time + 2)
            return self.post(url, payload)

        if isinstance(j, Dict) and "error" in j:
            error = j['error']
            if error == "overloaded":
                logger.warning(
                    "Waiting for overloaded API before trying again."
                )
                sleep(10)
                return self.post(url, payload)

            raise RuntimeError(error)

        assert isinstance(
            j, Sequence
        ) and len(j) > 0, f"Post did not return a sequence: {j}"

        # Use `temporary`` to make sure it doesn't get compiled away.
        logger.debug("I have allocated %s bytes.", sys.getsizeof(temporary))

        return j[0]


EndpointCallback.model_rebuild()
Endpoint.model_rebuild()
DummyEndpoint.model_rebuild()

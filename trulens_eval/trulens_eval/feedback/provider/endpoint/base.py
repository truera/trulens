from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import functools
import inspect
import logging
from pprint import PrettyPrinter
from queue import Queue
import random
import sys
from threading import Thread
from time import sleep
from types import ModuleType
from typing import (
    Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Type,
    TypeVar
)
import warnings

import pydantic
import requests

from trulens_eval.schema import Cost
from trulens_eval.utils.asynchro import desync
from trulens_eval.utils.asynchro import sync
from trulens_eval.utils.asynchro import ThunkMaybeAwaitable
from trulens_eval.utils.pyschema import safe_getattr
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.python import get_first_local_in_call_stack
from trulens_eval.utils.python import is_really_coroutinefunction
from trulens_eval.utils.python import locals_except
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import SingletonPerName
from trulens_eval.utils.python import Thunk
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import SerialModel
from trulens_eval.utils.threading import DEFAULT_NETWORK_TIMEOUT

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

T = TypeVar("T")

INSTRUMENT = "__tru_instrument"
DEFAULT_RPM = 60


class EndpointCallback(SerialModel):
    """
    Callbacks to be invoked after various API requests and track various metrics
    like token usage.
    """

    cost: Cost = pydantic.Field(default_factory=Cost)

    def handle(self, response: Any) -> None:
        self.cost.n_requests += 1

    def handle_chunk(self, response: Any) -> None:
        self.cost.n_stream_chunks += 1

    def handle_generation(self, response: Any) -> None:
        self.handle(response)

    def handle_generation_chunk(self, response: Any) -> None:
        self.handle_chunk(response)

    def handle_classification(self, response: Any) -> None:
        self.handle(response)


class Endpoint(WithClassInfo, SerialModel, SingletonPerName):

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    @dataclass
    class EndpointSetup():
        """
        Class for storing supported endpoint information. See `atrack_all_costs`
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
        )
    ]

    # Dict of classe/module-methods that have been instrumented for cost
    # tracking along with the wrapper methods and the class that instrumented
    # them. Key is the class or module owning the instrumented method. Tuple
    # value has:
    # - original function,
    # - wrapped version,
    # - endpoint that did the wrapping.
    instrumented_methods: ClassVar[Dict[Any, List[Tuple[Callable, Callable, Type[Endpoint]]]]] \
        = defaultdict(list) # pydantic.Field([], exclude=True)

    # API/endpoint name
    name: str

    # Requests per minute.
    rpm: float = DEFAULT_RPM

    # Retries (if performing requests using this class). TODO: wire this up to
    # the various endpoint systems' retries specification.
    retries: int = 3

    # Optional post headers for post requests if done by this class.
    post_headers: Dict[str, str] = pydantic.Field(
        default_factory=dict, exclude=True
    )

    # Queue that gets filled at rate rpm.
    pace: Queue = pydantic.Field(
        default_factory=lambda: Queue(maxsize=10), exclude=True
    )

    # Track costs not run inside "atrack_cost" here. Also note that Endpoints
    # are singletons (one for each unique name argument) hence this global
    # callback will track all requests for the named api even if you try to
    # create multiple endpoints (with the same name).
    global_callback: EndpointCallback = pydantic.Field(
        exclude=True
    )  # of type _callback_class

    # Callback class to use for usage tracking
    callback_class: Type[EndpointCallback] = pydantic.Field(exclude=True)

    # Name of variable that stores the callback noted above.
    callback_name: str = pydantic.Field(exclude=True)

    # Thread that fills the queue at the appropriate rate.
    pace_thread: Thread = pydantic.Field(exclude=True)

    def __new__(cls, *args, name: Optional[str] = None, **kwargs):
        name = name or cls.__name__
        return super().__new__(cls, *args, name=name, **kwargs)

    def __init__(self, *args, name: str, callback_class: Any = None, **kwargs):
        """
        API usage, pacing, and utilities for API endpoints.

        - `callback_class` should be set by subclass.
        """

        if safe_hasattr(self, "rpm"):
            # already initialized via the SingletonPerName mechanism
            return

        if callback_class is None:
            raise ValueError(
                "Endpoint has to be extended by class that can set `callback_class`."
            )

        kwargs['name'] = name
        kwargs['callback_class'] = callback_class
        kwargs['global_callback'] = callback_class()
        kwargs['callback_name'] = f"callback_{name}"
        kwargs['pace_thread'] = Thread()  # temporary
        kwargs['pace_thread'].daemon = True
        super().__init__(*args, **kwargs)

        def keep_pace():
            while True:
                sleep(60.0 / self.rpm)
                self.pace.put(True)

        self.pace_thread = Thread(target=keep_pace)
        self.pace_thread.daemon = True
        self.pace_thread.start()

        logger.debug(f"*** Creating {self.name} endpoint ***")

        # Extending class should call _instrument_module on the appropriate
        # modules and methods names.

    def pace_me(self):
        """
        Block until we can make a request to this endpoint.
        """

        self.pace.get()

        return

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
            logger.error(f"Waiting for {j} ({wait_time}) second(s).")
            sleep(wait_time + 2)
            return self.post(url, payload)

        if isinstance(j, Dict) and "error" in j:
            error = j['error']
            logger.error(f"API error: {j}.")
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

    def run_me(self, thunk: Thunk[T]) -> T:
        """
        Run the given thunk, returning itse output, on pace with the api.
        Retries request multiple times if self.retries > 0.
        """

        retries = self.retries + 1
        retry_delay = 2.0

        while retries > 0:
            try:
                self.pace_me()
                ret = thunk()
                return ret
            except Exception as e:
                retries -= 1
                logger.error(
                    f"{self.name} request failed {type(e)}={e}. Retries remaining={retries}."
                )
                if retries > 0:
                    sleep(retry_delay)
                    retry_delay *= 2

        raise RuntimeError(
            f"API {self.name} request failed {self.retries+1} time(s)."
        )

    def _instrument_module(self, mod: ModuleType, method_name: str) -> None:
        if safe_hasattr(mod, method_name):
            logger.debug(
                f"Instrumenting {mod.__name__}.{method_name} for {self.name}"
            )
            func = getattr(mod, method_name)
            w = self.wrap_function(func)

            # setattr(w, INSTRUMENT, func) # mark the new method indicating it is our wrapper

            setattr(mod, method_name, w)

            Endpoint.instrumented_methods[mod].append((func, w, type(self)))

    def _instrument_class(self, cls, method_name: str) -> None:
        if safe_hasattr(cls, method_name):
            logger.debug(
                f"Instrumenting {cls.__name__}.{method_name} for {self.name}"
            )
            func = getattr(cls, method_name)
            w = self.wrap_function(func)

            # setattr(w, INSTRUMENT, func) # mark the new method indicating it is our wrapper

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
                wrapped_thing if wrapped_thing !=
                object else "unknown dynamically generated class(es)"
            )
            for original, wrapped, endpoint in wrappers:
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
                f"Instrumenting method creator {cls.__name__}.{wrapper_method_name} "
                f"for {self.name}"
            )
            func = getattr(cls, wrapper_method_name)

            def metawrap(*args, **kwargs):

                produced_func = func(*args, **kwargs)

                if wrapped_method_filter(produced_func):

                    logger.debug(f"Instrumenting {produced_func.__name__}")

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
            logger.debug(f"module {mod} already instrumented for {method_name}")
            return

        for m in dir(mod):
            logger.debug(
                f"instrumenting module {mod} member {m} for method {method_name}"
            )
            if safe_hasattr(mod, m):
                obj = safe_getattr(mod, m)
                self._instrument_class(obj, method_name=method_name)

        already_instrumented.add(method_name)

    @staticmethod
    async def atrack_all_costs(
        thunk: ThunkMaybeAwaitable[T],
        with_openai: bool = True,
        with_hugs: bool = True,
        with_litellm: bool = True,
        with_bedrock: bool = True,
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
                        "Could not initiallize endpoint {cls.__name__}. "
                        "Possibly missing key(s). "
                        f"trulens_eval will not track costs/usage of this endpoint. {e}"
                    )

        return await Endpoint._atrack_costs(thunk, with_endpoints=endpoints)

    @staticmethod
    async def atrack_all_costs_tally(
        thunk: ThunkMaybeAwaitable[T],
        with_openai: bool = True,
        with_hugs: bool = True,
        with_litellm: bool = True,
        with_bedrock: bool = True,
    ) -> Tuple[T, Cost]:
        """
        Track costs of all of the apis we can currently track, over the
        execution of thunk.
        """

        result, cbs = await Endpoint.atrack_all_costs(
            thunk,
            with_openai=with_openai,
            with_hugs=with_hugs,
            with_litellm=with_litellm,
            with_bedrock=with_bedrock,
        )

        if len(cbs) == 0:
            # Otherwise sum returns "0" below.
            costs = Cost()
        else:
            costs = sum(cb.cost for cb in cbs)

        return result, costs

    @staticmethod
    async def _atrack_costs(
        thunk: ThunkMaybeAwaitable[T],
        with_endpoints: Optional[Sequence[Endpoint]] = None,
    ) -> Tuple[T, Sequence[EndpointCallback]]:
        """
        Root of all cost tracking methods. Runs the given `thunk`, tracking
        costs using each of the provided endpoints' callbacks.
        """

        # Check to see if this call is within another _atrack_costs call:
        endpoints: Dict[Type[EndpointCallback], Sequence[Tuple[Endpoint, EndpointCallback]]] = \
            get_first_local_in_call_stack(
                key="endpoints",
                func=Endpoint.__find_tracker,
                offset=1
            )

        if endpoints is None:
            # If not, lets start a new collection of endpoints here along with
            # the callbacks for each. See type above.

            endpoints = dict()

        else:
            # We copy the dict here so that the outer call to _track_costs will
            # have their own version unaffacted by our additions below. Once
            # this frame returns, the outer frame will have its own endpoints
            # again and any wrapped method will get that smaller set of
            # endpoints.

            # TODO: check if deep copy is needed given we are storing lists in
            # the values and don't want to affect the existing ones here.
            endpoints = endpoints.copy()

        # Collect any new endpoints requested of us.
        with_endpoints = with_endpoints or []

        # Keep track of the new callback objects we create here for returning
        # later.
        callbacks = []

        # Create the callbacks for the new requested endpoints only. Existing
        # endpoints from other frames will keep their callbacks.
        for endpoint in with_endpoints:
            callback_class = endpoint.callback_class
            callback = callback_class()

            if callback_class not in endpoints:
                endpoints[callback_class] = []

            # And add them to the endpoints dict. This will be retrieved from
            # locals of this frame later in the wrapped methods.
            endpoints[callback_class].append((endpoint, callback))

            callbacks.append(callback)

        # Call the thunk and wait for result.
        result: T = await desync(thunk)

        # Return result and only the callbacks created here. Outer thunks might
        # return others.
        return result, callbacks

    async def atrack_cost(
        self, thunk: ThunkMaybeAwaitable[T]
    ) -> Tuple[T, EndpointCallback]:
        """
        Tally only the usage performed within the execution of the given thunk.
        Returns the thunk's result alongside the EndpointCallback object that
        includes the usage information.
        """

        result, callbacks = await Endpoint._atrack_costs(
            thunk, with_endpoints=[self]
        )

        return result, callbacks[0]

    @staticmethod
    def __find_tracker(f):
        return id(f) == Endpoint._atrack_costs.__code__

    def handle_wrapped_call(
        self, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:
        """
        This gets called with the results of every instrumented method. This
        should be implemented by each subclass.

        Args:

        - func: Callable -- the wrapped function which returned.

        - bindings: BoundArguments -- the inputs to the wrapped method.

        - response: Any -- whatever the wrapped function returned.

        - callback: Optional[EndpointCallback] -- the callback set up by
          `atrack_cost` if the wrapped method was called and returned within an
          invocation of `atrack_cost`.
        """
        pass

    def wrap_function(self, func):
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
                    f"{func.__name__} already instrumented for callbacks of type {self.callback_class.__name__}"
                )

                return func

            else:
                # Otherwise add our callback class but don't instrument again.

                registered_callback_classes += [self.callback_class]
                setattr(func, INSTRUMENT, registered_callback_classes)

                return func

        # If INSTRUMENT is not set, create a wrapper method and return it.
        @functools.wraps(func)
        async def awrapper(*args, **kwargs):
            logger.debug(
                f"Calling wrapped async {func.__name__} for {self.name}, "
                f"iscoroutinefunction={is_really_coroutinefunction(func)}, "
                f"isasyncgenfunction={inspect.isasyncgenfunction(func)}"
            )

            # Get the result of the wrapped function:
            response = await desync(func, *args, **kwargs)

            bindings = inspect.signature(func).bind(*args, **kwargs)

            # Get all of the callback classes suitable for handling this call.
            # Note that we stored this in the INSTRUMENT attribute of the
            # wrapper method.
            registered_callback_classes = getattr(awrapper, INSTRUMENT)

            # Look up the endpoints that are expecting to be notified and the
            # callback tracking the tally. See Endpoint._atrack_costs for
            # definition.
            endpoints: Dict[Type[EndpointCallback], Sequence[Tuple[Endpoint, EndpointCallback]]] = \
                get_first_local_in_call_stack(
                    key="endpoints",
                    func=self.__find_tracker,
                    offset=0
                )

            # If wrapped method was not called from within _atrack_costs, we
            # will get None here and do nothing but return wrapped function's
            # response.
            if endpoints is None:
                logger.debug("No endpoints found.")
                return response

            for callback_class in registered_callback_classes:
                logger.debug(f"Handling callback_class: {callback_class}.")
                if callback_class not in endpoints:
                    logger.warning(
                        f"Callback class {callback_class.__name__} is registered for handling {func.__name__}"
                        " but there are no endpoints waiting to receive the result."
                    )
                    continue

                for endpoint, callback in endpoints[callback_class]:
                    logger.debug(f"Handling endpoint {endpoint}.")
                    endpoint.handle_wrapped_call(
                        func=func,
                        bindings=bindings,
                        response=response,
                        callback=callback
                    )

            return response

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(
                f"Calling instrumented sync method {func} of type {type(func)}, "
                f"iscoroutinefunction={is_really_coroutinefunction(func)}, "
                f"isasyncgeneratorfunction={inspect.isasyncgenfunction(func)}"
            )
            return sync(awrapper, *args, **kwargs)

        for w in [wrapper, awrapper]:
            # Set our tracking attribute to tell whether something is already
            # instrumented onto both the sync and async version since either one
            # could be returned from this method.
            setattr(w, INSTRUMENT, [self.callback_class])

        # Determine which of the wrapper variants to return and to annotate.

        if is_really_coroutinefunction(func):
            w = awrapper
        else:
            w = wrapper

        logger.debug(f"Instrumenting {func.__name__} for {self.name} .")

        return w


class DummyEndpoint(Endpoint):
    """
    Endpoint for testing purposes. Should not make any network calls.
    """

    # Simulated result parameters below.

    # How often to produce the "model loading" response.
    loading_prob: float
    # How much time to indicate as needed to load the model in the above response.
    loading_time: Callable[[], float] = \
        pydantic.Field(exclude=True, default_factory=lambda: lambda: random.uniform(0.73, 3.7))

    # How often to produce an error response.
    error_prob: float

    # How often to freeze instead of producing a response.
    freeze_prob: float

    # How often to produce the overloaded message.
    overloaded_prob: float

    # How much data in bytes to allocate when making requests.
    alloc: int

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
        rpm: float = DEFAULT_RPM * 10,
        **kwargs
    ):
        if safe_hasattr(self, "callback_class"):
            # Already created with SingletonPerName mechanism
            return

        assert error_prob + freeze_prob + overloaded_prob + loading_prob <= 1.0
        assert rpm > 0

        kwargs['name'] = name
        kwargs['callback_class'] = EndpointCallback

        super().__init__(
            **kwargs, **locals_except("self", "name", "kwargs", "__class__")
        )

        logger.info(
            f"Using DummyEndpoint with {locals_except('self', 'name', 'kwargs', '__class__')}"
        )

    def post(
        self,
        url: str,
        payload: JSON,
        timeout: float = DEFAULT_NETWORK_TIMEOUT
    ) -> Any:
        # Classification results only, like from huggingface. Simulates
        # overloaded, model loading, frozen, error.

        self.pace_me()

        # pretend to do this:
        """
        ret = requests.post(
            url, json=payload, timeout=timeout, headers=self.post_headers
        )
        """

        # allocate some data to pretend we are doing hard work
        temporary = [0x42] * self.alloc

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

            j = dict(estimated_time=self.loading_time())
        r -= self.loading_prob

        if r < self.overloaded_prob:
            # Simulated overloaded outcome.

            j = dict(error="overloaded")
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
            warnings.warn(
                f"Waiting for {j} ({wait_time}) second(s).",
                ResourceWarning,
                stacklevel=2
            )
            sleep(wait_time + 2)
            return self.post(url, payload)

        if isinstance(j, Dict) and "error" in j:
            error = j['error']
            if error == "overloaded":
                warnings.warn(
                    "Waiting for overloaded API before trying again.",
                    ResourceWarning,
                    stacklevel=2
                )
                sleep(10)
                return self.post(url, payload)
            else:
                raise RuntimeError(error)

        assert isinstance(
            j, Sequence
        ) and len(j) > 0, f"Post did not return a sequence: {j}"

        # Use `temporary`` to make sure it doesn't get compiled away.
        logger.debug(f"I have allocated {sys.getsizeof(temporary)} bytes.")

        return j[0]

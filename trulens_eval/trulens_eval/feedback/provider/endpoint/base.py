from __future__ import annotations

from collections import defaultdict
import inspect
import logging
from pprint import PrettyPrinter
from time import sleep
from types import ModuleType
from typing import (
    Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Type,
    TypeVar
)

from pydantic import Field
import requests

from trulens_eval import trace as mod_trace
from trulens_eval.schema import base as base_schema
from trulens_eval.utils import pace as pace_utils
from trulens_eval.utils import pyschema as pyschema_utils
from trulens_eval.utils import python as python_utils
from trulens_eval.utils import serial as serial_utils
from trulens_eval.utils import threading as threading_utils
from trulens_eval.utils import wrap as wrap_utils

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")

INSTRUMENT = "__tru_instrument"

DEFAULT_RPM = 60
"""Default requests per minute for endpoints."""


class EndpointCallback(mod_trace.TracingCallbacks[T]):
    """Extension to TracingCallbacks that tracks costs."""

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

        logger.warning("No on_response method defined for %s.", self)


#    def on_chunk(self, response: Any) -> None:
#        """Called after receiving a chunk from a request."""
#        self.cost.n_stream_chunks += 1

# our optional

    def on_endpoint_generation(self, response: Any) -> None:
        """Called after each completion request."""

        self.cost.n_successful_requests += 1
        self.cost.n_generations += 1

        # self.on_request(response)

    # our optional
    def on_endpoint_generation_chunk(self, response: Any) -> None:
        """Called after receiving a chunk from a completion request."""

        self.cost.n_stream_chunks += 1

    # our optional
    def on_endpoint_classification(self, response: Any) -> None:
        """Called after each classification response."""

        self.cost.n_successful_requests += 1
        self.cost.n_classifications += 1

        # logger.warning("No on_classification method defined for %s.", self)


class Endpoint(pyschema_utils.WithClassInfo, serial_utils.SerialModel,
               python_utils.SingletonPerName):
    """API usage, pacing, and utilities for API endpoints."""

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

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

    pace: pace_utils.Pace = Field(
        default_factory=lambda: pace_utils.
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
        if python_utils.safe_hasattr(self, "rpm"):
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
        kwargs['pace'] = pace_utils.Pace(
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

    @staticmethod
    def track_all_costs_tally(func: Callable[..., T], *args,
                              **kwargs) -> Tuple[T, base_schema.Cost]:
        """Track the costs of all instrumented methods in the given thunk.

        Args:
            func: The function to run and track costs for.

            args: The arguments to pass to the function.

            kwargs: The keyword arguments to pass to the function.

        Returns:
            The return value of func and the total cost of all cost instrumented
            methods called during the execution of func.
        """

        tracer = mod_trace.get_tracer()
        ret = None
        with tracer.phantom() as span:
            ret = func(*args, **kwargs)

        return ret, span.total_cost()

    def post(
        self,
        url: str,
        payload: serial_utils.JSON,
        timeout: float = threading_utils.DEFAULT_NETWORK_TIMEOUT
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

    def run_me(self, thunk: python_utils.Thunk[T]) -> T:
        """
        DEPRECTED: Run the given thunk, returning itse output, on pace with the api.
        Retries request multiple times if self.retries > 0.

        DEPRECATED: Use `run_in_pace` instead.
        """

        raise NotImplementedError(
            "This method is deprecated. Use `run_in_pace` instead."
        )

    def _instrument_module(self, mod: ModuleType, method_name: str) -> None:
        if python_utils.safe_hasattr(mod, method_name):
            logger.debug(
                "Instrumenting %s.%s for %s", python_utils.module_name(mod),
                method_name, self.name
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

        if python_utils.safe_hasattr(cls, method_name):
            logger.debug(
                "Instrumenting %s.%s for %s", python_utils.class_name(cls),
                method_name, self.name
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
                wrapped_thing if wrapped_thing !=
                object else "unknown dynamically generated class(es)"
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
        if python_utils.safe_hasattr(cls, wrapper_method_name):
            logger.debug(
                "Instrumenting method creator %s.%s for %s", cls.__name__,
                wrapper_method_name, self.name
            )
            func = getattr(cls, wrapper_method_name)

            def metawrap(*args, **kwargs):

                produced_func = func(*args, **kwargs)

                if wrapped_method_filter(produced_func):

                    logger.debug(
                        "Instrumenting %s",
                        python_utils.callable_name(produced_func)
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

        if not python_utils.safe_hasattr(mod, INSTRUMENT):
            setattr(mod, INSTRUMENT, set())

        already_instrumented = pyschema_utils.getattr_serial(mod, INSTRUMENT)

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
            if python_utils.safe_hasattr(mod, m):
                obj = pyschema_utils.getattr_serial(mod, m)
                if inspect.isclass(obj):
                    self._instrument_class(obj, method_name=method_name)

        already_instrumented.add(method_name)

    def wrap_function(self, func: Callable):
        """Create a wrapper of the given function to perform cost tracking.
        
        Args:
            func: The function to wrap.
        """

        return wrap_utils.wrap_callable(
            func=func,
            func_name=python_utils.callable_name(func),
            callback_class=self.callback_class,
            endpoint=self
        )


# EndpointCallback.model_rebuild()
Endpoint.model_rebuild()

import inspect
import json
import logging
from pprint import PrettyPrinter
from queue import Queue
from threading import Thread
from time import sleep
from types import ModuleType
from typing import (
    Any, Callable, Dict, Optional, Sequence, Tuple, Type, TypeVar
)

from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.schema import LLMResult
import pydantic
import requests

from trulens_eval.keys import _check_key
from trulens_eval.keys import get_huggingface_headers
from trulens_eval.schema import Cost
from trulens_eval.util import get_local_in_call_stack
from trulens_eval.util import JSON
from trulens_eval.util import SerialModel
from trulens_eval.util import SingletonPerName
from trulens_eval.util import WithClassInfo
from trulens_eval.utils.text import UNICODE_CHECK

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

    def handle_generation(self, response: Any) -> None:
        self.handle(response)

    def handle_classification(self, response: Any) -> None:
        self.handle(response)


class HuggingfaceCallback(EndpointCallback):

    def handle_classification(self, response: requests.Response) -> None:
        # Huggingface free inference api doesn't seem to have its own library
        # and the docs say to use `requests`` so that is what we instrument and
        # process to track api calls.

        super().handle_classification(response)

        if response.ok:
            self.cost.n_successful_requests += 1
            content = json.loads(response.text)

            # Huggingface free inference api for classification returns a list
            # with one element which itself contains scores for each class.
            self.cost.n_classes += len(content[0])


class OpenAICallback(EndpointCallback):

    class Config:
        arbitrary_types_allowed = True

    # For openai cost tracking, we use the logic from langchain mostly
    # implemented in the OpenAICallbackHandler class:
    langchain_handler: OpenAICallbackHandler = pydantic.Field(
        default_factory=OpenAICallbackHandler, exclude=True
    )

    def handle_classification(self, response: Dict) -> None:
        # OpenAI's moderation API is not text generation and does not return
        # usage information. Will count those as a classification.

        super().handle_classification(response)

        if "categories" in response:
            self.cost.n_successful_requests += 1
            self.cost.n_classes += len(response['categories'])

    def handle_generation(self, response: LLMResult) -> None:

        super().handle_generation(response)

        self.langchain_handler.on_llm_end(response)

        # Copy over the langchain handler fields we also have.
        for cost_field, langchain_field in [
            ("cost", "total_cost"), ("n_tokens", "total_tokens"),
            ("n_successful_requests", "successful_requests"),
            ("n_prompt_tokens", "prompt_tokens"),
            ("n_completion_tokens", "completion_tokens")
        ]:
            setattr(
                self.cost, cost_field,
                getattr(self.langchain_handler, langchain_field)
            )


class Endpoint(SerialModel, SingletonPerName):

    class Config:
        arbitrary_types_allowed = True

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

    # Track costs not run inside "track_cost" here. Also note that Endpoints are
    # singletons (one for each unique name argument) hence this global callback
    # will track all requests for the named api even if you try to create
    # multiple endpoints (with the same name).
    global_callback: EndpointCallback = pydantic.Field(
        exclude=True
    )  # of type _callback_class

    # Callback class to use for usage tracking
    callback_class: Type[EndpointCallback] = pydantic.Field(exclude=True)

    # Name of variable that stores the callback noted above.
    callback_name: str = pydantic.Field(exclude=True)

    # Thread that fills the queue at the appropriate rate.
    pace_thread: Thread = pydantic.Field(exclude=True)

    def __new__(cls, name: str, *args, **kwargs):
        return super(SingletonPerName, cls).__new__(
            SerialModel, name=name, *args, **kwargs
        )

    def __init__(self, *args, name: str, callback_class: Any, **kwargs):
        """
        API usage, pacing, and utilities for API endpoints.
        """

        if hasattr(self, "rpm"):
            # already initialized via the SingletonPerName mechanism
            return

        kwargs['name'] = name
        kwargs['callback_class'] = callback_class
        kwargs['global_callback'] = callback_class()
        kwargs['callback_name'] = f"callback_{name}"
        kwargs['pace_thread'] = Thread()  # temporary

        super(SerialModel, self).__init__(*args, **kwargs)

        def keep_pace():
            while True:
                sleep(60.0 / self.rpm)
                self.pace.put(True)

        self.pace_thread = Thread(target=keep_pace)
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
        self, url: str, payload: JSON, timeout: Optional[int] = None
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

        return j[0]

    def run_me(self, thunk: Callable[[], T]) -> T:
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
        if hasattr(mod, method_name):
            logger.debug(
                f"Instrumenting {mod.__name__}.{method_name} for {self.name}"
            )
            func = getattr(mod, method_name)
            w = self.wrap_function(func)
            setattr(mod, method_name, w)

    def _instrument_class(self, cls, method_name: str) -> None:
        if hasattr(cls, method_name):
            logger.debug(
                f"Instrumenting {cls.__name__}.{method_name} for {self.name}"
            )
            func = getattr(cls, method_name)
            w = self.wrap_function(func)
            setattr(cls, method_name, w)

    def _instrument_module_members(self, mod: ModuleType, method_name: str):
        logger.debug(
            f"Instrumenting {mod.__package__}.*.{method_name} for {self.name}"
        )

        for m in dir(mod):
            obj = getattr(mod, m)
            self._instrument_class(obj, method_name=method_name)

    @staticmethod
    def track_all_costs(
        thunk: Callable[[], T],
        with_openai: bool = True,
        with_hugs: bool = True
    ) -> Tuple[T, Sequence[EndpointCallback]]:
        """
        Track costs of all of the apis we can currently track, over the
        execution of thunk.
        """

        endpoints = []

        if with_openai:
            try:
                e = OpenAIEndpoint()
                endpoints.append(e)
            except:
                logger.warning(
                    "OpenAI API keys are not set. "
                    "Will not track usage."
                )

        if with_hugs:
            try:
                e = HuggingfaceEndpoint()
                endpoints.append(e)
            except:
                logger.warning(
                    "Huggingface API keys are not set. "
                    "Will not track usage."
                )

        return Endpoint._track_costs(thunk, with_endpoints=endpoints)

    @staticmethod
    def track_all_costs_tally(
        thunk: Callable[[], T],
        with_openai: bool = True,
        with_hugs: bool = True
    ) -> Tuple[T, Cost]:
        """
        Track costs of all of the apis we can currently track, over the
        execution of thunk.
        """

        result, cbs = Endpoint.track_all_costs(
            thunk, with_openai=with_openai, with_hugs=with_hugs
        )
        return result, sum(cb.cost for cb in cbs)

    @staticmethod
    def _track_costs(
        thunk: Callable[[], T],
        with_endpoints: Sequence['Endpoint'] = None,
    ) -> Tuple[T, Sequence[EndpointCallback]]:
        """
        Root of all cost tracking methods. Runs the given `thunk`, tracking
        costs using each of the provided endpoints' callbacks.
        """

        # Check to see if this call is within another _track_costs call:
        endpoints: Dict[Type[EndpointCallback], Sequence[Tuple[Endpoint, EndpointCallback]]] = \
            get_local_in_call_stack(
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

        # Call the thunk.
        result: T = thunk()

        # Return result and only the callbacks created here. Outer thunks might
        # return others.
        return result, callbacks

    def track_cost(self, thunk: Callable[[], T]) -> Tuple[T, EndpointCallback]:
        """
        Tally only the usage performed within the execution of the given thunk.
        Returns the thunk's result alongside the EndpointCallback object that
        includes the usage information.
        """

        result, callbacks = Endpoint._track_costs(thunk, with_endpoints=[self])

        return result, callbacks[0]

    @staticmethod
    def __find_tracker(f):
        return id(f) == id(Endpoint._track_costs.__code__)

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
          `track_cost` if the wrapped method was called and returned within an
          invocation of `track_cost`.
        """
        pass

    def wrap_function(self, func):
        if hasattr(func, INSTRUMENT):
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

        def wrapper(*args, **kwargs):
            logger.debug(f"Calling wrapped {func.__name__} for {self.name}.")

            # Get the result of the wrapped function:
            response: Any = func(*args, **kwargs)

            bindings = inspect.signature(func).bind(*args, **kwargs)

            # Get all of the callback classes suitable for handling this call.
            # Note that we stored this in the INSTRUMENT attribute of the
            # wrapper method.
            registered_callback_classes = getattr(wrapper, INSTRUMENT)

            # Look up the endpoints that are expecting to be notified and the
            # callback tracking the tally. See Endpoint._track_costs for
            # definition.
            endpoints: Dict[Type[EndpointCallback], Sequence[Tuple[Endpoint, EndpointCallback]]] = \
                get_local_in_call_stack(
                    key="endpoints",
                    func=self.__find_tracker,
                    offset=0
                )

            # If wrapped method was not called from within _track_costs, we will
            # get None here and do nothing but return wrapped function's
            # response.
            if endpoints is None:
                return response

            for callback_class in registered_callback_classes:
                if callback_class not in endpoints:
                    logger.warning(
                        f"Callback class {callback_class.__name__} is registered for handling {func.__name__}"
                        " but there are no endpoints waiting to receive the result."
                    )
                    continue

                for endpoint, callback in endpoints[callback_class]:

                    endpoint.handle_wrapped_call(
                        func=func,
                        bindings=bindings,
                        response=response,
                        callback=callback
                    )

            return response

        setattr(wrapper, INSTRUMENT, [self.callback_class])
        wrapper.__name__ = func.__name__
        wrapper.__signature__ = inspect.signature(func)

        logger.debug(f"Instrumenting {func.__name__} for {self.name} .")

        return wrapper


class OpenAIEndpoint(Endpoint, WithClassInfo):
    """
    OpenAI endpoint. Instruments "create" methods in openai.* classes.
    """

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="openai")

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:

        model_name = ""
        if 'model' in bindings.kwargs:
            model_name = bindings.kwargs['model']

        results = None
        if "results" in response:
            results = response['results']

        counted_something = False

        if 'usage' in response:
            counted_something = True
            usage = response['usage']

            llm_res = LLMResult(
                generations=[[]],
                llm_output=dict(token_usage=usage, model_name=model_name),
                run=None
            )

            self.global_callback.handle_generation(response=llm_res)

            if callback is not None:
                callback.handle_generation(response=llm_res)

        if results is not None:
            for res in results:
                if "categories" in res:
                    counted_something = True
                    self.global_callback.handle_classification(response=res)

                    if callback is not None:
                        callback.handle_classification(response=res)

        if not counted_something:
            logger.warning(
                f"Unregonized openai response format. It did not have usage information nor categories:\n"
                + pp.pformat(response)
            )

    def __init__(self, *args, **kwargs):
        # If any of these keys are in kwargs, copy over its value to the env
        # variable named as the respective value in this dict. If value is None,
        # don't copy to env. Regardless of env, set all of these as attributes
        # to openai.

        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/switching-endpoints
        CONF_CLONE = dict(
            api_key="OPENAI_API_KEY",
            organization=None,
            api_type=None,
            api_base=None,
            api_version=None
        )

        import os

        import openai

        for k, v in CONF_CLONE.items():
            if k in kwargs:
                print(f"{UNICODE_CHECK} Setting openai.{k} explicitly.")
                setattr(openai, k, kwargs[k])

                if v is not None:
                    print(f"{UNICODE_CHECK} Env. var. {v} set explicitly.")
                    os.environ[v] = kwargs[k]
            else:
                if v is not None:
                    # If no value were explicitly set, check if the user set up openai
                    # attributes themselves and if so, copy over the ones we use via
                    # environment vars, to its respective env var.

                    attr_val = getattr(openai, k)
                    if attr_val is not None and attr_val != os.environ.get(v):
                        print(
                            f"{UNICODE_CHECK} Env. var. {v} set from openai.{k} ."
                        )
                        os.environ[v] = attr_val

        # Will fail if key not set:
        _check_key("OPENAI_API_KEY")

        if hasattr(self, "name"):
            # Already created with SingletonPerName mechanism
            return

        kwargs['name'] = "openai"
        kwargs['callback_class'] = OpenAICallback

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)

        self._instrument_module_members(openai, "create")


class HuggingfaceEndpoint(Endpoint, WithClassInfo):
    """
    Huggingface. Instruments the requests.post method for requests to
    "https://api-inference.huggingface.co".
    """

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="huggingface")

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments,
        response: requests.Response, callback: Optional[EndpointCallback]
    ) -> None:
        # Call here can only be requests.post .

        if "url" not in bindings.arguments:
            return

        url = bindings.arguments['url']
        if not url.startswith("https://api-inference.huggingface.co"):
            return

        # TODO: Determine whether the request was a classification or some other
        # type of request. Currently we use huggingface only for classification
        # in feedback but this can change.

        self.global_callback.handle_classification(response=response)

        if callback is not None:
            callback.handle_classification(response=response)

    def __init__(self, *args, **kwargs):
        # Will fail if key not set:
        _check_key("HUGGINGFACE_API_KEY")

        if hasattr(self, "name"):
            # Already created with SingletonPerName mechanism
            return

        kwargs['name'] = "huggingface"
        kwargs['post_headers'] = get_huggingface_headers()
        kwargs['callback_class'] = HuggingfaceCallback

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)

        self._instrument_class(requests, "post")

from abc import ABC, abstractmethod
import inspect
import json
import logging
# from multiprocessing import Queue
from queue import Queue
from threading import Thread
from time import sleep
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, TypeVar
import pydantic

import requests

from trulens_eval.db import JSON
from trulens_eval.schema import Cost
from trulens_eval.keys import get_huggingface_headers
from trulens_eval.util import SerialModel, get_local_in_call_stack
from trulens_eval.util import SingletonPerName
from trulens_eval.util import TP

from langchain.schema import LLMResult
from langchain.callbacks.openai_info import OpenAICallbackHandler

logger = logging.getLogger(__name__)

T = TypeVar("T")

INSTRUMENT = "__tru_instrument"

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
        arbitrary_types_allowed=True

    langchain_handler: OpenAICallbackHandler = pydantic.Field(
        default_factory=OpenAICallbackHandler,
        exclude=True
    )

    def handle_generation(self, response: LLMResult) -> None:
        # For openai we use the mechanisms from langchain. 

        super().handle_generation(response)

        self.langchain_handler.on_llm_end(response)

        # Copy over the langchain handler fields we also have.
        for cost_field, langchain_field in [
            ("cost", "total_cost"),
            ("n_tokens", "total_tokens"),
            ("n_successful_requests", "successful_requests"),
            ("n_prompt_tokens", "prompt_tokens"),
            ("n_completion_tokens", "completion_tokens")
        ]:
            setattr(self.cost, cost_field, getattr(self.langchain_handler, langchain_field))
        

class Endpoint(SerialModel, SingletonPerName):#, ABC):
    class Config:
        arbitrary_types_allowed = True
        # underscore_attrs_are_private = False

    # API/endpoint name
    name: str

    # Requests per minute.
    rpm: float = 60

    # Retries (if performing requests using this class). TODO: wire this up to
    # the various endpoint systems' retries specification.
    retries: int = 3

    # Optional post headers for post requests if done by this class.
    post_headers: Dict[str, str] = pydantic.Field(default_factory=dict, exclude=True)

    # Queue that gets filled at rate rpm.
    pace: Queue = pydantic.Field(default_factory=lambda: Queue(maxsize=10), exclude=True)

    # Track costs not run inside "track_cost" here. Also note that Endpoints are
    # singletons (one for each unique name argument) hence this global callback
    # will track all requests for the named api even if you try to create
    # multiple endpoints (with the same name).
    global_callback: EndpointCallback = pydantic.Field(exclude=True) # of type _callback_class

    # Callback class to use for usage tracking
    callback_class: Type[EndpointCallback] = pydantic.Field(exclude=True)

    # Name of variable that stores the callback noted above.
    callback_name: str = pydantic.Field(exclude=True)

    # Thread that fills the queue at the appropriate rate.    
    pace_thread: Thread = pydantic.Field(exclude=True)

    # TODO: validate to construct tracking objects when deserializing?

    def __new__(cls, name: str, *args, **kwargs):
        return super(SingletonPerName, cls).__new__(SerialModel, name=name, *args, **kwargs)

    def __init__(
        self, *args, name: str, callback_class: Any, **kwargs
    ):
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
        kwargs['pace_thread'] = Thread() # temporary

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
        ret = requests.post(url, json=payload, timeout=timeout, headers=self.post_headers)

        j = ret.json()

        # Huggingface public api sometimes tells us that a model is loading and
        # how long to wait:
        if "estimated_time" in j:
            wait_time = j['estimated_time']
            logger.error(f"Waiting for {j} ({wait_time}) second(s).")
            sleep(wait_time + 2)
            return self.post(url, payload)

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
                    f"{self.name} request failed {type(e)}={e}. Retries={retries}."
                )
                if retries > 0:
                    sleep(retry_delay)
                    retry_delay *= 2

        raise RuntimeError(
            f"API {self.name} request failed {self.retries+1} time(s)."
        )

    def _instrument_module(self, mod: ModuleType, method_name: str) -> None:
        if hasattr(mod, method_name):
            logger.debug(f"Instrumenting {mod.__name__}.{method_name} for {self.name}")
            func = getattr(mod, method_name)  
            w = self.wrap_function(func)
            setattr(mod, method_name, w)

    def _instrument_class(self, cls, method_name: str) -> None:
        if hasattr(cls, method_name):
            logger.debug(f"Instrumenting {cls.__name__}.{method_name} for {self.name}")
            func = getattr(cls, method_name)  
            w = self.wrap_function(func)
            setattr(cls, method_name, w)


    def _instrument_module_members(self, mod: ModuleType, method_name: str):
        logger.debug(f"Instrumenting {mod.__package__}.*.{method_name} for {self.name}")

        for m in dir(mod):
            obj = getattr(mod, m)
            self._instrument_class(obj, method_name=method_name)


    def track_cost(self, thunk: Callable[[], T]) -> Tuple[T, EndpointCallback]:
        """
        Tally only the usage performed within the execution of the given thunk.
        Returns the thunk's result alongside the EndpointCallback object that
        includes the usage information.
        """

        # Keep this here for access by wrappers higher in call stack.
        cb = self.callback_class()
        locals()[self.callback_name] = cb

        return thunk(), cb

    @staticmethod
    def __find_tracker(f):
        return id(f) == id(Endpoint.track_cost.__code__)

    # @abstractmethod
    def handle_wrapped_call(
        self,
        bindings: inspect.BoundArguments,
        response: Any,
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
            # TODO: What if we want to instrument the same method for multiple
            # endpoints?
            logger.debug(f"{func.__name__} already instrumented")
            return func

        def wrapper(*args, **kwargs):
            logger.debug(f"Calling wrapped {func.__name__} for {self.name}.")
            
            res = func(*args, **kwargs)

            cb = get_local_in_call_stack(
                key=self.callback_name,
                func=self.__find_tracker,
                offset=0
            )

            bindings = inspect.signature(func).bind(*args, **kwargs)

            self.handle_wrapped_call(func=func, bindings=bindings, response=res, callback=cb)
            
            return res
        
        setattr(wrapper, INSTRUMENT, func)
        wrapper.__name__ = func.__name__
        wrapper.__signature__ = inspect.signature(func)

        logger.debug(f"Instrumenting {func.__name__} for {self.name} .")

        return wrapper


class OpenAIEndpoint(Endpoint):
    """
    OpenAI endpoint. Instruments "create" methods in openai.* classes.
    """

    def __new__(cls):
        return super(Endpoint, cls).__new__(cls, name="openai")

    def handle_wrapped_call(
        self,
        func: Callable, 
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:

        model_name = None
        if 'model' in bindings.kwargs:
            model_name = bindings.kwargs['model']

        usage = None
        if 'usage' in response:
            usage = response['usage']

        llm_res = LLMResult(
            generations=[[]],
            llm_output=dict(token_usage=usage, model_name=model_name),
            run=None
        )

        self.global_callback.handle_generation(response=llm_res)
        if callback is not None:
            callback.handle_generation(response=llm_res)

    def __init__(self, *args, **kwargs):
        kwargs['name'] = "openai"
        kwargs['callback_class'] = OpenAICallback

        super().__init__(*args, **kwargs)

        import openai
        self._instrument_module_members(openai, "create")


class HuggingfaceEndpoint(Endpoint):
    """
    OpenAI endpoint. Instruments "create" methodsin openai.* classes.
    """

    def __new__(cls):
        return super(Endpoint, cls).__new__(cls, name="huggingface")

    def handle_wrapped_call(
        self,
        func: Callable, 
        bindings: inspect.BoundArguments,
        response: requests.Response,
        callback: Optional[EndpointCallback]
    ) -> None:
    
        self.global_callback.handle_classification(response=response)

        if callback is not None:
            callback.handle_classification(response=response)

    def __init__(self, *args, **kwargs):
        kwargs['name'] = "huggingface"
        kwargs['post_headers'] = get_huggingface_headers()
        kwargs['callback_class'] = HuggingfaceCallback

        super().__init__(*args, **kwargs)

        self._instrument_class(requests, "post")

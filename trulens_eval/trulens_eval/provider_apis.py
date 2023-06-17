import inspect
import logging
from multiprocessing import Queue
# from queue import Queue
from threading import Thread
from time import sleep
from types import ModuleType
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar
import pydantic

import requests

from trulens_eval.db import JSON
from trulens_eval.util import SerialModel, get_local_in_call_stack
from trulens_eval.util import SingletonPerName
from trulens_eval.util import TP

from langchain.schema import LLMResult
from langchain.callbacks.openai_info import OpenAICallbackHandler

logger = logging.getLogger(__name__)

T = TypeVar("T")

INSTRUMENT = "__tru_instrument"

class Endpoint(SerialModel, SingletonPerName):
    class Config:
        pass

    # API/endpoint name
    name: str

    # Requests per minute.
    rpm: float = 60

    # Retries (if performing requests using this class).
    retries: int = 3

    # Optional post headers for post requests if done by this class.
    post_headers: Optional[Any] = None

    # Queue that will  get filled at rate rpm.
    _pace: Queue = pydantic.Field(default_factory=lambda: Queue(maxsize=10))

    # Track costs not run inside "track_cost" here.
    _global_callback: Any # of type _callback_class

    # Callback class to use for usage tracking
    _callback_class: type

    # Name of variable that stores the callback noted above.
    _callback_name: str

    # Thread that fills the queue at the appropriate rate.    
    _pace_thread: Thread = None

    def __init__(
        self, *args, name: str, callback_class: type, **kwargs
    ):
        """
        API usage, pacing, and utilities for API endpoints.
        """

        if hasattr(self, "rpm"):
            # already initialized via the SingletonPerName mechanism
            return

        kwargs['name'] = name
        kwargs['_callback_class'] = callback_class
        kwargs['_global_callback'] = callback_class()
        kwargs['_callback_name'] = f"callback_{name}"

        super().__init__(*args, **kwargs)

        logger.debug(f"*** Creating {self.name} endpoint ***")

        self._start_pace()
        
        # Extending class should call _instrument_module on the appropriate
        # modules and methods names.

    def pace_me(self):
        """
        Block until we can make a request to this endpoint.
        """

        self._pace.get()

        return

    def post(
        self, url: str, payload: JSON, timeout: Optional[int] = None
    ) -> Any:
        extra = dict()
        if self.post_headers is not None:
            extra['headers'] = self.post_headers

        self.pace_me()
        ret = requests.post(url, json=payload, timeout=timeout, **extra)

        j = ret.json()

        # Huggingface public api sometimes tells us that a model is loading and how long to wait:
        if "estimated_time" in j:
            wait_time = j['estimated_time']
            logger.error(f"Waiting for {j} ({wait_time}) second(s).")
            sleep(wait_time + 2)
            return self.post(url, payload)

        assert isinstance(
            j, Sequence
        ) and len(j) > 0, f"Post did not return a sequence: {j}"

        return j[0]

    def run_me(self, thunk):
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

    def _start_pace(self):

        def keep_pace():
            while True:
                sleep(60.0 / self.rpm)
                self._pace.put(True)

        thread = Thread(target=keep_pace)
        thread.start()

        self._pace_thread = thread

    def _instrument_module(self, mod: ModuleType, method_name: str):
        for m in dir(mod):
            sub = getattr(mod, m)
            
            if hasattr(sub, method_name):
                logger.debug(f"Instrumenting {mod.__package__}.{method_name} for {self.name}")
                func = getattr(sub, method_name)  
                w = self.wrap_function(func)
                setattr(sub, method_name, w)

    def track_cost(self, thunk: Callable[[], T]) -> Tuple[T, Any]: # Any -> langchain llm callback handler
        """
        Tally only the openai API usage performed within the execution of the
        given thunk. Returns the thunk's result alongside the langchain callback
        that includes the usage information.
        """

        # Keep this here for access by wrappers higher in call stack.
        cb = self.callback_class()
        locals()[self.callback_name] = cb

        return thunk(), cb

    @staticmethod
    def __find_tracker(f):
        return id(f) == id(Endpoint.track_cost.__code__)

    def wrap_function(self, func):
        if hasattr(func, INSTRUMENT):
            # TODO: What if we want to instrument the same method for multiple
            # endpoints?
            logger.debug(f"{func.__name__} already instrumented")
            return func

        def wrapper(*args, **kwargs):
            logger.debug(f"Calling wrapped {func.__name__} for {self.endpoint_name}.")
            
            res = func(*args, **kwargs)

            model_name = None
            if 'model' in kwargs:
                model_name = kwargs['model']

            usage = None
            if 'usage' in res:
                usage = res['usage']

            llm_res = LLMResult(
                generations=[[]],
                llm_output=dict(token_usage=usage, model_name=model_name),
                run=None
            )

            cb = get_local_in_call_stack(
                key=self.callback_name,
                func=self.__find_tracker,
                offset=0
            )

            self.global_callback.on_llm_end(response=llm_res)
            
            if cb is not None:
                cb.on_llm_end(response=llm_res)
    
            return res
        
        setattr(wrapper, INSTRUMENT, func)
        wrapper.__name__ = func.__name__
        wrapper.__signature__ = inspect.signature(func)

        logger.debug(f"Instrumenting {func.__name__} for {self.endpoint_name} .")

        return wrapper


class OpenAIEndpoint(Endpoint):
    """
    Track openai uses. This makes use of langchain OpenAICallbackHandler for
    extracting and tallying various openai API response content.
    """

    def __init__(self, *args, **kwargs):
        kwargs['name'] = "openai"
        kwargs['_callback_class'] = OpenAICallbackHandler

        super().__init__(*args, **kwargs)

        import openai

        self._instrument_module(openai, "create")
"""
Dummy API and Endpoint.

These are are meant to resemble (make similar sequences of calls) real APIs and
Endpoints but not they do not actually make any network requests. Some
randomness is introduced to simulate the behavior of real APIs.
"""

from __future__ import annotations

import inspect
import logging
from pprint import pformat
import random
from time import sleep
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

from numpy import random as np_random
import numpy as np
import pydantic
from pydantic import Field

from trulens_eval.feedback.provider.endpoint.base import DEFAULT_RPM
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.feedback.provider.endpoint.base import INSTRUMENT
from trulens_eval.utils.python import locals_except
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.threading import DEFAULT_NETWORK_TIMEOUT

logger = logging.getLogger(__name__)

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")


class NonDeterminism(pydantic.BaseModel):
    """Hold random number generators and seeds for controlling non-deterministic behavior."""

    random: Any = Field(exclude=True)
    """Random number generator."""

    np_random: Any = Field(exclude=True)
    """Numpy Random number generator."""

    seed: int = 0xdeadbeef
    """Control randomness."""

    def __init__(self, **kwargs):
        kwargs['random'] = None
        kwargs['np_random'] = None

        super().__init__(**kwargs)

        self.random = random.Random(self.seed)
        self.np_random = np_random.RandomState(self.seed)

    def discrete_choice(self, seq: Sequence[A], probs: Sequence[float]) -> A:
        """Sample a random element from a sequence with the given
        probabilities."""

        return self.random.choices(seq, weights=probs, k=1)[0]


class DummyAPI(pydantic.BaseModel):
    """A dummy model evaluation API used by DummyEndpoint.

    This is meant to stand in for classes such as OpenAI.completion . Methods in
    this class are instrumented for cost tracking testing.
    """

    loading_time_uniform_params: Tuple[pydantic.NonNegativeFloat,
                                       pydantic.NonNegativeFloat] = (0.7, 3.7)
    """How much time to indicate as needed to load the model.
    
    Parameters of a uniform distribution.
    """

    loading_prob: pydantic.NonNegativeFloat = 0.0
    """How often to produce the "model loading" response that huggingface api
    sometimes produces."""

    error_prob: pydantic.NonNegativeFloat = 0.0
    """How often to produce an error response."""

    freeze_prob: pydantic.NonNegativeFloat = 0.0
    """How often to freeze instead of producing a response."""

    overloaded_prob: pydantic.NonNegativeFloat = 0.0
    """How often to produce the overloaded message that huggingface sometimes
    produces."""

    # NOTE: All probability not covered by the above is for a normal response
    # without error or delay.

    alloc: pydantic.NonNegativeInt = 1024
    """How much data in bytes to allocate when making requests."""

    delay: pydantic.NonNegativeFloat = 0.0
    """How long to delay each request.
    
    Delay is normally distributed with this mean and half this standard
    deviation, in seconds. Any delay sample below 0 is replaced with 0.
    """

    ndt: NonDeterminism = Field(default_factory=NonDeterminism)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.error_prob + self.freeze_prob + self.overloaded_prob + self.loading_prob <= 1.0, \
            "Total probabilites should not exceed 1.0 ."

    def post(
        self, url: str, payload: JSON, timeout: Optional[float] = None
    ) -> Any:
        """Pretend to make an http post request to some model execution API."""

        assert isinstance(payload, dict), "Payload should be a dict."

        if timeout is None:
            timeout = DEFAULT_NETWORK_TIMEOUT

        # allocate some data to pretend we are doing hard work
        temporary = np.empty(self.alloc, dtype=np.int8)

        if self.delay > 0.0:
            sleep(
                max(0.0, self.ndt.np_random.normal(self.delay, self.delay / 2))
            )

        r = self.ndt.discrete_choice(
            seq=["normal", "freeze", "error", "loading", "overloaded"],
            probs=[
                1 - self.freeze_prob - self.error_prob - self.loading_prob -
                self.overloaded_prob, self.freeze_prob, self.error_prob,
                self.loading_prob, self.overloaded_prob
            ]
        )

        if r == "freeze":
            # Simulated freeze outcome.

            while True:
                sleep(timeout)

        elif r == "error":
            # Simulated error outcome.

            raise RuntimeError("Simulated error happened.")

        elif r == "loading":
            # Simulated loading model outcome.

            wait_time = self.ndt.np_random.uniform(
                *self.wait_time_uniform_params
            )
            logger.warning(
                "Waiting for model to load (%s) second(s).",
                wait_time,
            )
            sleep(wait_time + 2)
            return self.post(url, payload, timeout=timeout)

        elif r == "overloaded":
            # Simulated overloaded outcome.

            logger.warning("Waiting for overloaded API before trying again.")
            sleep(10)
            return self.post(url, payload, timeout=timeout)

        elif r == "normal":

            if "api-inference.huggingface.co" in url:
                # pretend to produce huggingface api classification results
                return self._fake_classification()
            else:
                return self._fake_completion(
                    model=payload['model'],
                    prompt=payload['prompt'],
                    temperature=payload['temperature']
                )

        else:
            raise RuntimeError("Unknown random result type.")

    def _fake_completion(
        self, model: str, prompt: str, temperature: float
    ) -> Dict:
        generated_text: str = f"my original response to model {model} with temperature {temperature} is {prompt}"

        return {
            'completion': generated_text,
            'status': 'success',
            'usage':
                {
                    # Fake usage information.
                    'n_tokens':
                        len(generated_text.split()) + len(prompt.split()),
                    'n_prompt_tokens':
                        len(prompt.split()),
                    'n_completion_tokens':
                        len(generated_text.split()),
                    'cost':
                        len(generated_text) * 0.0002 +
                        len(prompt.split()) * 0.0001
                }
        }

    def completion(
        self, *args, model: str, temperature: float = 0.0, prompt: str
    ) -> Dict:
        """Fake text completion request."""

        # Fake http post request, might raise an exception or cause delays.
        return self.post(
            url="https://fakeservice.com/classify",
            payload={
                'mode': 'completion',
                'model': model,
                'prompt': prompt,
                'temperature': temperature,
                'args': args  # include extra args to see them in post span
            }
        )

    def _fake_classification(self):
        # Simulated success outcome with some random scores. Should add up to 1.
        r1 = self.ndt.random.uniform(0.1, 0.9)
        r2 = self.ndt.random.uniform(0.1, 0.9 - r1)
        r3 = 1 - (r1 + r2)

        return [
            {
                'label': 'LABEL_1',
                'score': r1
            }, {
                'label': 'LABEL_2',
                'score': r2
            }, {
                'label': 'LABEL_0',
                'score': r3
            }
        ]

    def classification(
        self, *args, model: str = "fakeclassier", text: str
    ) -> Dict:
        """Fake classification request."""

        # Fake http post request, might raise an exception or cause delays.
        return self.post(
            url=
            "https://api-inference.huggingface.co/classify",  # url makes the fake post produce fake classification scores
            payload={
                'mode': 'classification',
                'model': model,
                'inputs': text,
                'args': args  # include extra args to see them in post span
            }
        )


class DummyAPICreator():
    """Creator of DummyAPI methods.
    
    This is used for testing instrumentation of classes like
    `boto3.ClientCreator`.
    """

    def __init__(self, *args, **kwargs):
        self.api_args = args
        self.api_kwargs = kwargs

    def create_method(self, method_name: str) -> DummyAPI:
        """Dynamically create a method that behaves like a DummyAPI method.
        
        This method should be instrumented by `DummyEndpoint` for testing method
        creation like that of `boto3.ClientCreator._create_api_method`.
        """

        # Creates a new class just to make things harder for ourselves.
        class DynamicDummyAPI(DummyAPI):
            pass

        return getattr(
            DynamicDummyAPI(*self.api_args, **self.api_kwargs), method_name
        )


class DummyEndpointCallback(EndpointCallback):
    """Callbacks for instrumented methods in DummyAPI to recover costs from those calls."""

    def handle_classification(self, response: Sequence) -> None:
        super().handle_classification(response)

        if "scores" in response:
            # fake classification
            self.cost.n_classes += len(response)

    def handle_generation(self, response: Dict) -> None:
        super().handle_generation(response=response)

        if "usage" in response:
            # fake completion
            usage = response["usage"]
            self.cost.cost += usage.get("cost", 0.0)
            self.cost.n_tokens += usage.get("n_tokens", 0)
            self.cost.n_prompt_tokens += usage.get("n_prompt_tokens", 0)
            self.cost.n_completion_tokens += usage.get("n_completion_tokens", 0)


class DummyEndpoint(Endpoint):
    """Endpoint for testing purposes.
    
    Does not make any network calls and just pretends to.
    """

    api: DummyAPI = Field(default_factory=DummyAPI)
    """Fake API to use for making fake requests."""

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="dummyendpoint")

    def __init__(
        self,
        name: str = "dummyendpoint",
        rpm: float = DEFAULT_RPM * 10,
        **kwargs
    ):
        if safe_hasattr(self, "callback_class"):
            # Already created with SingletonPerName mechanism
            return

        assert rpm > 0

        kwargs['name'] = name
        kwargs['callback_class'] = DummyEndpointCallback

        kwargs['api'] = DummyAPI(**kwargs)
        # Will use fake api for fake feedback evals.

        super().__init__(
            **kwargs, **locals_except("self", "name", "kwargs", "__class__")
        )

        logger.info(
            "Using DummyEndpoint with %s",
            locals_except('self', 'name', 'kwargs', '__class__')
        )

        # Instrument existing DummyAPI class. These are used by the custom_app
        # example.
        self._instrument_class(DummyAPI, "completion")
        self._instrument_class(DummyAPI, "classify")

        # Also instrument any dynamically created DummyAPI methods like we do
        # for boto3.ClientCreator.
        if not safe_hasattr(DummyAPICreator.create_method, INSTRUMENT):
            self._instrument_class_wrapper(
                DummyAPICreator,
                wrapper_method_name="create_method",
                wrapped_method_filter=lambda f: f.__name__ in
                ['completion', 'classify']
            )

    def post(
        self, url: str, payload: JSON, timeout: Optional[float] = None
    ) -> Dict:
        return self.api.post(url, payload, timeout=timeout)

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[EndpointCallback],
    ) -> None:
        logger.debug(
            "Handling dummyapi instrumented call to func: %s,\n"
            "\tbindings: %s,\n"
            "\tresponse: %s", func, bindings, response
        )

        if "usage" in response:
            counted_something = True
            self.global_callback.handle_generation(response=response)

            if callback is not None:
                callback.handle_generation(response=response)

        elif isinstance(response, Sequence):
            counted_something = True
            self.global_callback.handle_classification(response=response)

            if callback is not None:
                callback.handle_classification(response=response)

        if not counted_something:
            logger.warning(
                "Could not find usage information in DummyAPI response:\n%s",
                pformat(response)
            )

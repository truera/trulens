"""Dummy API and Endpoint.

These are are meant to resemble (make similar sequences of calls) real APIs and
Endpoints but not they do not actually make any network requests. Some
randomness is introduced to simulate the behavior of real APIs.
"""

from __future__ import annotations

import asyncio
from enum import Enum
import inspect
import io
import json as mod_json
import logging
from pprint import pformat
import random
from time import sleep
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
from numpy import random as np_random
import pydantic
from pydantic import Field
import requests
from requests import structures as request_structures
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import threading as threading_utils

logger = logging.getLogger(__name__)

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")


class _DummyOutcome(Enum):
    """Outcomes of a dummy API call."""

    NORMAL = "normal"
    """Normal response."""

    FREEZE = "freeze"
    """Simulated freeze outcome."""

    ERROR = "error"
    """Simulated error outcome."""

    LOADING = "loading"
    """Simulated loading model outcome."""

    OVERLOADED = "overloaded"
    """Simulated overloaded outcome."""


class NonDeterminism(pydantic.BaseModel):
    """Hold random number generators and seeds for controlling non-deterministic behavior."""

    random: Any = Field(exclude=True)
    """Random number generator."""

    np_random: Any = Field(exclude=True)
    """Numpy Random number generator."""

    seed: int = 0xDEADBEEF
    """Control randomness."""

    def __init__(self, **kwargs):
        kwargs["random"] = None
        kwargs["np_random"] = None

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

    loading_time_uniform_params: Tuple[
        pydantic.NonNegativeFloat, pydantic.NonNegativeFloat
    ] = (0.7, 3.7)
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

    ndt: NonDeterminism = Field(default_factory=NonDeterminism, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            self.error_prob
            + self.freeze_prob
            + self.overloaded_prob
            + self.loading_prob
            <= 1.0
        ), "Total probabilities should not exceed 1.0 ."

    def _fake_post_request(
        self, url: str, json: serial_utils.JSON, headers: Optional[Dict] = None
    ) -> requests.Request:
        """Fake requests.Request object for a fake post request."""

        return requests.Request(
            method="POST",
            url=url,
            headers=headers,
            data=json,
        )

    def _fake_post_response(
        self,
        status_code: int,
        json: serial_utils.JSON,
        request: requests.Request,
    ) -> requests.Response:
        """Fake requests.Response object for a fake post request."""

        res = requests.Response()

        res.status_code = status_code

        res._content = mod_json.dumps(json).encode()
        res._content_consumed = True
        res.raw = io.BytesIO(res._content)  # might not be needed

        res.headers = request_structures.CaseInsensitiveDict({
            "content-type": "application/json"
        })

        res.request = request.prepare()

        return res

    async def apost(
        self,
        url: str,
        json: serial_utils.JSON,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> requests.Response:
        """Pretend to make an http post request to some model execution API."""

        assert isinstance(json, dict), "json should be a dict."

        if timeout is None:
            timeout = threading_utils.DEFAULT_NETWORK_TIMEOUT

        if headers is None:
            headers = {}

        request = self._fake_post_request(url, json=json, headers=headers)

        # allocate some data to pretend we are doing hard work
        temporary = np.empty(self.alloc, dtype=np.int8)  # noqa: F841

        if self.delay > 0.0:
            await asyncio.sleep(
                max(0.0, self.ndt.np_random.normal(self.delay, self.delay / 2))
            )

        r = self.ndt.discrete_choice(
            seq=list(_DummyOutcome),
            probs=[
                1
                - self.freeze_prob
                - self.error_prob
                - self.loading_prob
                - self.overloaded_prob,
                self.freeze_prob,
                self.error_prob,
                self.loading_prob,
                self.overloaded_prob,
            ],
        )

        if r == _DummyOutcome.FREEZE:
            # Simulated freeze outcome.

            while True:
                await asyncio.sleep(timeout)

        elif r == _DummyOutcome.ERROR:
            # Simulated error outcome.

            raise RuntimeError("Simulated error happened.")

        elif r == _DummyOutcome.LOADING:
            # Simulated loading model outcome.

            wait_time = self.ndt.np_random.uniform(
                *self.loading_time_uniform_params
            )
            logger.warning(
                "Waiting for model to load (%s) second(s).",
                wait_time,
            )
            await asyncio.sleep(wait_time + 2)
            return await self.apost(
                url=url, json=json, timeout=timeout, headers=headers
            )

        elif r == _DummyOutcome.OVERLOADED:
            # Simulated overloaded outcome.

            logger.warning("Waiting for overloaded API before trying again.")
            await asyncio.sleep(10)
            return await self.apost(
                url=url, json=json, timeout=timeout, headers=headers
            )

        elif r == _DummyOutcome.NORMAL:
            if "api-inference.huggingface.co" in url:
                # pretend to produce huggingface api classification results
                ret = self._fake_classification()
            else:
                ret = self._fake_completion(
                    model=json["model"],
                    prompt=json["prompt"],
                    temperature=json["temperature"],
                )

            return self._fake_post_response(
                status_code=200, json=ret, request=request
            )

        else:
            raise RuntimeError("Unknown random result type.")

    def post(
        self,
        url: str,
        json: serial_utils.JSON,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> requests.Response:
        """Pretend to make an http post request to some model execution API."""

        assert isinstance(json, dict), "Payload should be a dict."

        if headers is None:
            headers = {}

        if timeout is None:
            timeout = threading_utils.DEFAULT_NETWORK_TIMEOUT

        request = self._fake_post_request(url, json=json, headers=headers)

        # allocate some data to pretend we are doing hard work
        temporary = np.empty(self.alloc, dtype=np.int8)  # noqa: F841

        if self.delay > 0.0:
            sleep(
                max(0.0, self.ndt.np_random.normal(self.delay, self.delay / 2))
            )

        r = self.ndt.discrete_choice(
            seq=list(_DummyOutcome),
            probs=[
                1
                - self.freeze_prob
                - self.error_prob
                - self.loading_prob
                - self.overloaded_prob,
                self.freeze_prob,
                self.error_prob,
                self.loading_prob,
                self.overloaded_prob,
            ],
        )

        if r == _DummyOutcome.FREEZE:
            # Simulated freeze outcome.

            while True:
                sleep(timeout)

        elif r == _DummyOutcome.ERROR:
            # Simulated error outcome.

            raise RuntimeError("Simulated error happened.")

        elif r == _DummyOutcome.LOADING:
            # Simulated loading model outcome.

            wait_time = self.ndt.np_random.uniform(
                *self.loading_time_uniform_params
            )
            logger.warning(
                "Waiting for model to load (%s) second(s).",
                wait_time,
            )
            sleep(wait_time + 2)
            return self.post(url, json=json, timeout=timeout, headers=headers)

        elif r == _DummyOutcome.OVERLOADED:
            # Simulated overloaded outcome.

            logger.warning("Waiting for overloaded API before trying again.")
            sleep(10)
            return self.post(url, json=json, timeout=timeout, headers=headers)

        elif r == _DummyOutcome.NORMAL:
            if "api-inference.huggingface.co" in url:
                # pretend to produce huggingface api classification results
                ret = self._fake_classification()
            else:
                ret = self._fake_completion(
                    model=json["model"],
                    prompt=json["prompt"],
                    temperature=json["temperature"],
                )

            return self._fake_post_response(
                status_code=200, json=ret, request=request
            )

        else:
            raise RuntimeError("Unknown random result type.")

    def _fake_completion(
        self, model: str, prompt: str, temperature: float
    ) -> List[serial_utils.JSON]:
        generated_text: str = f"""
First an integer: 2 . Also, this is my response to a prompt of length
{len(prompt)} with a model {model} with temperature {temperature}. Also, here is
an integer in case this is being used as a score: 2
"""

        return [
            {
                "completion": generated_text,
                "status": "success",
                "usage": {
                    # Fake usage information.
                    "n_tokens": len(generated_text.split())
                    + len(prompt.split()),
                    "n_prompt_tokens": len(prompt.split()),
                    "n_completion_tokens": len(generated_text.split()),
                    "cost": len(generated_text) * 0.0002
                    + len(prompt.split()) * 0.0001,
                },
            }
        ]

    def completion(
        self, *args, model: str, temperature: float = 0.0, prompt: str
    ) -> serial_utils.JSON:
        """Fake text completion request."""

        # TODO: move this to provider

        # Fake http post request, might raise an exception or cause delays.
        res = self.post(
            url="https://fakeservice.com/completion",
            json={
                "mode": "completion",
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "args": args,  # include extra args to see them in post span
            },
        )
        if res.status_code != 200:
            raise RuntimeError(
                f"Unexpected status from http response: {res.status_code}"
            )

        return res.json()[0]

    async def acompletion(
        self, *args, model: str, temperature: float = 0.0, prompt: str
    ) -> serial_utils.JSON:
        """Fake text completion request."""

        # TODO: move this to provider

        # Fake http post request, might raise an exception or cause delays.
        res = await self.apost(
            url="https://fakeservice.com/completion",
            json={
                "mode": "completion",
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "args": args,  # include extra args to see them in post span
            },
        )

        if res.status_code != 200:
            raise RuntimeError(
                f"Unexpected status from http response: {res.status_code}"
            )

        return res.json()[0]

    def _fake_classification(self) -> List[List[serial_utils.JSON]]:
        """Fake classification response in the form returned by huggingface."""

        # Simulated success outcome with some random scores. Should add up to 1.
        r1 = self.ndt.random.uniform(0.1, 0.9)
        r2 = self.ndt.random.uniform(0.1, 0.9 - r1)
        r3 = 1 - (r1 + r2)

        return [
            [
                {"label": "LABEL_1", "score": r1},
                {"label": "LABEL_2", "score": r2},
                {"label": "LABEL_0", "score": r3},
            ]
        ]

    def classification(
        self, *args, model: str = "fakeclassier", text: str
    ) -> List[serial_utils.JSON]:
        """Fake classification request."""

        # TODO: move this to provider

        # Fake http post request, might raise an exception or cause delays.
        return self.post(
            url="https://api-inference.huggingface.co/classify",  # url makes the fake post produce fake classification scores
            json={
                "mode": "classification",
                "model": model,
                "inputs": text,
                "args": args,  # include extra args to see them in post span
            },
        ).json()[0]

    async def aclassification(
        self, *args, model: str = "fakeclassier", text: str
    ) -> List[serial_utils.JSON]:
        """Fake classification request."""

        # TODO: move this to provider

        # Fake http post request, might raise an exception or cause delays.
        return (
            await self.apost(
                url="https://api-inference.huggingface.co/classify",  # url makes the fake post produce fake classification scores
                json={
                    "mode": "classification",
                    "model": model,
                    "inputs": text,
                    "args": args,  # include extra args to see them in post span
                },
            )
        ).json()[0]


class DummyAPICreator:
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


class DummyEndpointCallback(core_endpoint.EndpointCallback):
    """Callbacks for instrumented methods in DummyAPI to recover costs from those calls."""

    def handle_classification(self, response: Sequence) -> None:
        super().handle_classification(response)

        if all("score" in curr for curr in response):
            # fake classification
            self.cost.n_classes += len(response)

            self.cost.n_successful_requests += 1

    def handle_generation(self, response: Dict) -> None:
        super().handle_generation(response=response)

        if "usage" in response:
            # fake completion
            usage = response["usage"]
            self.cost.cost += usage.get("cost", 0.0)
            self.cost.n_tokens += usage.get("n_tokens", 0)
            self.cost.n_prompt_tokens += usage.get("n_prompt_tokens", 0)
            self.cost.n_completion_tokens += usage.get("n_completion_tokens", 0)

            self.cost.n_successful_requests += 1


class DummyEndpoint(core_endpoint._WithPost, core_endpoint.Endpoint):
    """Endpoint for testing purposes.

    Does not make any network calls and just pretends to.
    """

    api: DummyAPI = Field(default_factory=DummyAPI)
    """Fake API to use for making fake requests."""

    @deprecation_utils.deprecated_property(
        "Use `DummyEndpoint.api.alloc` instead."
    )
    def alloc(self) -> int:
        return self.api.alloc

    @deprecation_utils.deprecated_property(
        "Use `DummyEndpoint.api.float` instead."
    )
    def delay(self) -> float:
        return self.api.delay

    @deprecation_utils.deprecated_property(
        "Use `DummyEndpoint.api.error_prob` instead."
    )
    def error_prob(self) -> float:
        return self.api.error_prob

    @deprecation_utils.deprecated_property(
        "Use `DummyEndpoint.api.freeze_prob` instead."
    )
    def freeze_prob(self) -> float:
        return self.api.freeze_prob

    @deprecation_utils.deprecated_property(
        "Use `DummyEndpoint.api.loading_prob` instead."
    )
    def loading_prob(self) -> float:
        return self.api.loading_prob

    @deprecation_utils.deprecated_property(
        "Use `DummyEndpoint.api.loading_time_uniform_params` instead."
    )
    def loading_time(self) -> Tuple[float, float]:
        return self.api.loading_time_uniform_params

    @deprecation_utils.deprecated_property(
        "Use `DummyEndpoint.api.overloaded_prob` instead."
    )
    def overloaded_prob(self) -> float:
        return self.api.overloaded_prob

    def __init__(
        self,
        name: str = "dummyendpoint",
        rpm: float = core_endpoint.DEFAULT_RPM * 10,
        **kwargs,
    ):
        assert rpm > 0

        kwargs["name"] = name
        kwargs["callback_class"] = DummyEndpointCallback

        kwargs["api"] = DummyAPI(**kwargs)
        # Will use fake api for fake feedback evals.

        super().__init__(
            **kwargs,
            **python_utils.locals_except("self", "name", "kwargs", "__class__"),
        )

        logger.info(
            "Using DummyEndpoint with %s",
            python_utils.locals_except("self", "name", "kwargs", "__class__"),
        )

        # Instrument existing DummyAPI class. These are used by the custom_app
        # example. Note that `completion` and `classification` use post or apost so
        # we should not instrument both `classification` and `post` as this would
        # double count costs.
        self._instrument_class(DummyAPI, "post")
        self._instrument_class(DummyAPI, "apost")

        # Also instrument any dynamically created DummyAPI methods like we do
        # for boto3.ClientCreator.
        if not python_utils.safe_hasattr(
            DummyAPICreator.create_method, core_endpoint.INSTRUMENT
        ):
            self._instrument_class_wrapper(
                DummyAPICreator,
                wrapper_method_name="create_method",
                wrapped_method_filter=lambda f: f.__name__ in ["post", "apost"],
            )

    # Overrides all of WithPost.post as we don't want to make an actual request.
    def post(
        self,
        url: str,
        json: serial_utils.JSON,
        timeout: Optional[float] = None,
    ) -> requests.Response:
        self.pace_me()  # need this as we are not using WithPost.post

        return self.api.post(
            url, json=json, timeout=timeout, headers=self.post_headers
        )

    # Overrides all of WithPost.apost as we don't want to make an actual request.
    async def apost(
        self,
        url: str,
        json: serial_utils.JSON,
        timeout: Optional[float] = None,
    ) -> requests.Response:
        await self.apace_me()  # need this as we are not using WithPost.apost

        return await self.api.apost(
            url, json=json, timeout=timeout, headers=self.post_headers
        )

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> Any:
        # TODELETE(otel_tracing). Delete once otel_tracing is no longer
        # experimental.

        response = mod_json.loads(response.text)[0]

        logger.debug(
            "Handling dummyapi instrumented call to func: %s,\n"
            "\tbindings: %s,\n"
            "\tresponse: %s",
            func,
            bindings,
            response,
        )

        counted_something = False

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
                pformat(response),
            )

            return response

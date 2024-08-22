"""
# Dev Notes

This class makes use of langchain's cost tracking for openai models. Changes to
the involved classes will need to be adapted here. The important classes are:

- `langchain.schema.LLMResult`
- `langchain.callbacks.openai_info.OpenAICallbackHandler`

## Changes for openai 1.0

- Previously we instrumented classes `openai.*` and their methods `create` and
  `acreate`. Now we instrument classes `openai.resources.*` and their `create`
  methods. We also instrument `openai.resources.chat.*` and their `create`. To
  be determined is the instrumentation of the other classes/modules under
  `openai.resources`.

- openai methods produce structured data instead of dicts now. langchain expects
  dicts so we convert them to dicts.

"""

import inspect
import logging
import pprint
from typing import Any, Callable, ClassVar, Dict, List, Optional, TypeVar, Union

from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.schema import Generation
from langchain.schema import LLMResult
import pydantic
from pydantic.v1 import BaseModel as v1BaseModel
from trulens.core.feedback import endpoint as base_endpoint
from trulens.core.utils.constants import CLASS_INFO
from trulens.core.utils.pace import Pace
from trulens.core.utils.pyschema import Class
from trulens.core.utils.python import safe_getattr
from trulens.core.utils.python import safe_hasattr
from trulens.core.utils.serial import SerialModel

import openai as oai
from openai import resources
from openai.resources import chat
from trulens_eval.schema import base as base_schema

T = TypeVar("T")

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class OpenAIClient(SerialModel):
    """
    A wrapper for openai clients.

    This class allows wrapped clients to be serialized into json. Does not
    serialize API key though. You can access openai.OpenAI under the `client`
    attribute. Any attributes not defined by this wrapper are looked up from the
    wrapped `client` so you should be able to use this instance as if it were an
    `openai.OpenAI` instance.
    """

    REDACTED_KEYS: ClassVar[List[str]] = ["api_key", "default_headers"]
    """Parameters of the OpenAI client that will not be serialized because they
    contain secrets."""

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    client: Union[oai.OpenAI, oai.AzureOpenAI] = pydantic.Field(exclude=True)
    """Deserialized representation."""

    client_cls: Class
    """Serialized representation class."""

    client_kwargs: dict
    """Serialized representation constructor arguments."""

    def __init__(
        self,
        client: Optional[Union[oai.OpenAI, oai.AzureOpenAI]] = None,
        client_cls: Optional[Class] = None,
        client_kwargs: Optional[dict] = None,
    ):
        if client_kwargs is not None:
            # Check if any of the keys which will be redacted when serializing
            # were set and give the user a warning about it.
            for rkey in OpenAIClient.REDACTED_KEYS:
                if rkey in client_kwargs:
                    logger.warning(
                        f"OpenAI parameter {rkey} is not serialized for DEFERRED feedback mode. "
                        f"If you are not using DEFERRED, you do not need to do anything. "
                        f"If you are using DEFERRED, try to specify this parameter through env variable or another mechanism."
                    )

        if client is None:
            if client_kwargs is None and client_cls is None:
                client = oai.OpenAI()

            elif client_kwargs is None or client_cls is None:
                raise ValueError(
                    "`client_kwargs` and `client_cls` are both needed to deserialize an openai.`OpenAI` client."
                )

            else:
                if isinstance(client_cls, dict):
                    # TODO: figure out proper pydantic way of doing these things. I
                    # don't think we should be required to parse args like this.
                    client_cls = Class.model_validate(client_cls)

                cls = client_cls.load()

                timeout = client_kwargs.get("timeout")
                if timeout is not None:
                    client_kwargs["timeout"] = oai.Timeout(**timeout)

                client = cls(**client_kwargs)

        if client_cls is None:
            assert client is not None

            client_class = type(client)

            # Recreate constructor arguments and store in this dict.
            client_kwargs = {}

            # Guess the constructor arguments based on signature of __new__.
            sig = inspect.signature(client_class.__init__)

            for k, _ in sig.parameters.items():
                if k in OpenAIClient.REDACTED_KEYS:
                    # Skip anything that might have the api_key in it.
                    # default_headers contains the api_key.
                    continue

                if safe_hasattr(client, k):
                    client_kwargs[k] = safe_getattr(client, k)

            # Create serializable class description.
            client_cls = Class.of_class(client_class)

        super().__init__(
            client=client, client_cls=client_cls, client_kwargs=client_kwargs
        )

    def __getattr__(self, k):
        # Pass through attribute lookups to `self.client`, the openai.OpenAI
        # instance.
        if safe_hasattr(self.client, k):
            return safe_getattr(self.client, k)

        raise AttributeError(
            f"No attribute {k} in wrapper OpenAiClient nor the wrapped OpenAI client."
        )


class WrapperOpenAICallback(base_endpoint.WrapperEndpointCallback[T]):
    """EXPERIMENTAL: otel-tracing

    Process the results of instrumented openai calls to extract cost information.
    """

    # WrapperEndpointCallback optional
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.langchain_handler: OpenAICallbackHandler = OpenAICallbackHandler()

        self.chunks: List[Any] = []

    # WrapperEndpointCallback optional
    def on_endpoint_response(
        self,
        response: Any,
    ) -> None:
        """Process a returned response from an openai call.

        As there are multiple types of calls being handled here, we need to make
        various checks to see what sort of data to process based on the call
        made."""

        # TODO: cleanup/refactor.

        func = self.func
        bindings = self.bindings

        assert bindings is not None

        logger.debug(
            "Handling openai instrumented call to func: %s,\n"
            "\tbindings: %s,\n"
            "\tresponse: %s",
            func,
            bindings,
            response,
        )

        model_name = ""
        if "model" in bindings.kwargs:
            model_name = bindings.kwargs["model"]

        if isinstance(response, oai.Stream):
            # NOTE(piotrm): Merely checking membership in these will exhaust internal
            # genertors or iterators which will break users' code. While we work
            # out something, I'm disabling any cost-tracking for these streams.
            logger.warning("Cannot track costs from a OpenAI Stream.")
            return

        if not isinstance(response, dict):
            logger.warning(
                "OpenAI response is not a dict, got %s", type(response)
            )
            return

        results = None
        if "results" in response:
            results = response["results"]

        counted_something = False
        if hasattr(response, "usage"):
            counted_something = True

            if isinstance(response.usage, pydantic.BaseModel):
                usage = response.usage.model_dump()
            elif isinstance(response.usage, pydantic.v1.BaseModel):
                usage = response.usage.dict()
            elif isinstance(response.usage, Dict):
                usage = response.usage
            else:
                usage = None

            # See how to construct in langchain.llms.openai.OpenAIChat._generate
            llm_res = LLMResult(
                generations=[[]],
                llm_output=dict(token_usage=usage, model_name=model_name),
                run=None,
            )

            self.on_endpoint_generation(response=llm_res)

        if "choices" in response and "delta" in response.choices[0]:
            # Streaming data.
            content = response.choices[0].delta.content

            gen = Generation(text=content or "", generation_info=response)
            self.on_endpoint_generation_chunk(gen)

            counted_something = True

        if results is not None:
            for res in results:
                if "categories" in res:
                    counted_something = True
                    self.on_endpoint_classification(response=res)

        if not counted_something:
            logger.warning(
                "Could not find usage information in openai response:\n%s",
                pp.pformat(response),
            )

        return

    # WrapperEndpointCallback optional
    def on_endpoint_generation_chunk(self, response: Any) -> None:
        """Process a generation chunk."""

        super().on_endpoint_generation_chunk(response=response)

        self.chunks.append(response)

        if response.choices[0].finish_reason == "stop":
            llm_result = LLMResult(
                llm_output=dict(token_usage=dict(), model_name=response.model),
                generations=[self.chunks],
            )
            self.chunks = []
            self.on_endpoint_generation(response=llm_result)

    # WrapperEndpointCallback optional
    def on_endpoint_generation(self, response: LLMResult) -> None:
        """Process a generation/completion."""

        super().on_endpoint_generation(response)

        self.langchain_handler.on_llm_end(response)

        assert self.cost is not None

        self.cost += base_schema.Cost(**{
            cost_field: getattr(self.langchain_handler, langchain_field, 0)
            for cost_field, langchain_field in [
                ("cost", "total_cost"),
                ("n_tokens", "total_tokens"),
                ("n_successful_requests", "successful_requests"),
                ("n_prompt_tokens", "prompt_tokens"),
                ("n_completion_tokens", "completion_tokens"),
            ]
        })


class OpenAICallback(base_endpoint.EndpointCallback):
    # TODEP: remove after EXPERIMENTAL: otel-tracing

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    langchain_handler: OpenAICallbackHandler = pydantic.Field(
        default_factory=OpenAICallbackHandler, exclude=True
    )

    chunks: List[Generation] = pydantic.Field(
        default_factory=list,
        exclude=True,
    )

    def handle_generation_chunk(self, response: Any) -> None:
        super().handle_generation_chunk(response=response)

        self.chunks.append(response)

        if response.choices[0].finish_reason == "stop":
            llm_result = LLMResult(
                llm_output=dict(token_usage=dict(), model_name=response.model),
                generations=[self.chunks],
            )
            self.chunks = []
            self.handle_generation(response=llm_result)

    def handle_generation(self, response: LLMResult) -> None:
        super().handle_generation(response)

        self.langchain_handler.on_llm_end(response)

        for cost_field, langchain_field in [
            ("cost", "total_cost"),
            ("n_tokens", "total_tokens"),
            ("n_successful_requests", "successful_requests"),
            ("n_prompt_tokens", "prompt_tokens"),
            ("n_completion_tokens", "completion_tokens"),
        ]:
            setattr(
                self.cost,
                cost_field,
                getattr(self.cost, cost_field, 0)
                + getattr(self.langchain_handler, langchain_field, 0),
            )


class OpenAIEndpoint(base_endpoint.Endpoint):
    """OpenAI endpoint.

    Instruments "create" methods in openai client.

    Args:
        client: openai client to use. If not provided, a new client will be
            created using the provided kwargs.

        **kwargs: arguments to constructor of a new OpenAI client if `client`
            not provided.

    """

    client: OpenAIClient

    def __init__(
        self,
        name: str = "openai",
        client: Optional[
            Union[oai.OpenAI, oai.AzureOpenAI, OpenAIClient]
        ] = None,
        rpm: Optional[int] = None,
        pace: Optional[Pace] = None,
        **kwargs: dict,
    ):
        if safe_hasattr(self, "name") and client is not None:
            # Already created with SingletonPerName mechanism
            if len(kwargs) != 0:
                logger.warning(
                    "OpenAIClient singleton already made, ignoring arguments %s",
                    kwargs,
                )
                self.warning()  # issue info about where the singleton was originally created
            return

        self_kwargs = {
            "name": name,  # for SingletonPerName
            "rpm": rpm,
            "pace": pace,
            **kwargs,
        }

        self_kwargs["callback_class"] = OpenAICallback
        self_kwargs["wrapper_callback_class"] = WrapperOpenAICallback

        if CLASS_INFO in kwargs:
            del kwargs[CLASS_INFO]

        if client is None:
            # Pass kwargs to client.
            client = oai.OpenAI(**kwargs)
            self_kwargs["client"] = OpenAIClient(client=client)

        else:
            if len(kwargs) != 0:
                logger.warning(
                    "Arguments %s are ignored as `client` was provided.",
                    list(kwargs.keys()),
                )

            # Convert openai client to our wrapper if needed.
            if not isinstance(client, OpenAIClient):
                assert isinstance(
                    client, (oai.OpenAI, oai.AzureOpenAI)
                ), "OpenAI client expected"

                client = OpenAIClient(client=client)

            self_kwargs["client"] = client

        # for pydantic.BaseModel
        super().__init__(**self_kwargs)

        # Instrument various methods for usage/cost tracking.

        self._instrument_module_members(resources, "create")
        self._instrument_module_members(chat, "create")

    def __new__(cls, *args, **kwargs):
        return super(base_endpoint.Endpoint, cls).__new__(cls, name="openai")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[base_endpoint.EndpointCallback],
    ) -> None:
        # TODEP: remove after EXPERIMENTAL: otel-tracing

        # TODO: cleanup/refactor. This method inspects the results of an
        # instrumented call made by an openai client. As there are multiple
        # types of calls being handled here, we need to make various checks to
        # see what sort of data to process based on the call made.

        logger.debug(
            "Handling openai instrumented call to func: %s,\n"
            "\tbindings: %s,\n"
            "\tresponse: %s",
            func,
            bindings,
            response,
        )

        model_name = ""
        if "model" in bindings.kwargs:
            model_name = bindings.kwargs["model"]

        if isinstance(response, oai.Stream):
            # NOTE(piotrm): Merely checking membership in these will exhaust internal
            # generators or iterators which will break users' code. While we work
            # out something, I'm disabling any cost-tracking for these streams.
            logger.warning("Cannot track costs from a OpenAI Stream.")
            return

        results = None
        if "results" in response:
            results = response["results"]

        counted_something = False
        if hasattr(response, "usage"):
            counted_something = True

            if isinstance(response.usage, pydantic.BaseModel):
                usage = response.usage.model_dump()
            elif isinstance(response.usage, v1BaseModel):
                usage = response.usage.dict()
            elif isinstance(response.usage, Dict):
                usage = response.usage
            else:
                usage = None

            # See how to construct in langchain.llms.openai.OpenAIChat._generate
            llm_res = LLMResult(
                generations=[[]],
                llm_output=dict(token_usage=usage, model_name=model_name),
                run=None,
            )

            self.global_callback.handle_generation(response=llm_res)

            if callback is not None:
                callback.handle_generation(response=llm_res)

        if "choices" in response and "delta" in response.choices[0]:
            # Streaming data.
            content = response.choices[0].delta.content

            gen = Generation(text=content or "", generation_info=response)
            self.global_callback.handle_generation_chunk(gen)
            if callback is not None:
                callback.handle_generation_chunk(gen)

            counted_something = True

        if results is not None:
            for res in results:
                if "categories" in res:
                    counted_something = True
                    self.global_callback.handle_classification(response=res)

                    if callback is not None:
                        callback.handle_classification(response=res)

        if not counted_something:
            logger.warning(
                "Could not find usage information in openai response:\n%s",
                pp.pformat(response),
            )

"""
# Dev Notes

This class makes use of langchain's cost tracking for openai models. Changes to
the involved classes will need to be adapted here. The important classes are:

- `langchain.schema.LLMResult`
- `langchain.callbacks.openai_info.OpenAICallbackHandler`

# Changes in openai v1

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
from typing import Any, Callable, List, Optional

from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.schema import Generation
from langchain.schema import LLMResult
import openai as oai
import pydantic

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import safe_getattr
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import SerialModel

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class OpenAIClient(SerialModel):
    """
    A wrapper for openai clients that allows them to be serialized into json.
    Does not serialize API key though. You can access openai.OpenAI under the
    `client` attribute. Any attributes not defined by this wrapper are looked up
    from the wrapped `client` so you should be able to use this instance as if
    it were an `openai.OpenAI` instance.
    """

    class Config:
        arbitrary_types_allowed = True

    # Deserialized representation.
    client: oai.OpenAI = pydantic.Field(exclude=True)

    # Serialized representation.
    client_cls: Class
    client_kwargs: dict

    def __init__(
        self,
        client: Optional[oai.OpenAI] = None,
        client_cls: Optional[Class] = None,
        client_kwargs: Optional[dict] = None
    ):
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
                    client_cls = Class(**client_cls)

                cls = client_cls.load()

                timeout = client_kwargs.get("timeout")
                if timeout is not None:
                    client_kwargs['timeout'] = oai.Timeout(**timeout)

                client = cls(**client_kwargs)

        if client_cls is None:
            assert client is not None

            client_class = type(client)

            # Recreate constructor arguments and store in this dict.
            client_kwargs = {}

            # Guess the contructor arguments based on signature of __new__.
            sig = inspect.signature(client_class.__init__)

            for k, _ in sig.parameters.items():

                if k in ['api_key', 'default_headers']:
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


class OpenAICallback(EndpointCallback):

    class Config:
        arbitrary_types_allowed = True

    langchain_handler: OpenAICallbackHandler = pydantic.Field(
        default_factory=OpenAICallbackHandler, exclude=True
    )

    chunks: List[Generation] = pydantic.Field(
        default_factory=list, exclude=True
    )

    def handle_generation_chunk(self, response: Any) -> None:
        super().handle_generation_chunk(response=response)

        self.chunks.append(response)

        if response.choices[0].finish_reason == 'stop':
            llm_result = LLMResult(
                llm_output=dict(token_usage=dict(), model_name=response.model),
                generations=[self.chunks]
            )
            self.chunks = []
            self.handle_generation(response=llm_result)

    def handle_generation(self, response: LLMResult) -> None:
        super().handle_generation(response)

        self.langchain_handler.on_llm_end(response)

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


class OpenAIEndpoint(Endpoint, WithClassInfo):
    """
    OpenAI endpoint. Instruments "create" methods in openai client.
    """

    client: OpenAIClient

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="openai")

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:
        # TODO: cleanup/refactor. This method inspects the results of an
        # instrumented call made by an openai client. As there are multiple
        # types of calls being handled here, we need to make various checks to
        # see what sort of data to process based on the call made.

        logger.debug(
            f"Handling openai instrumented call to func: {func},\n"
            f"\tbindings: {bindings},\n"
            f"\tresponse: {response}"
        )

        model_name = ""
        if 'model' in bindings.kwargs:
            model_name = bindings.kwargs['model']

        results = None
        if "results" in response:
            results = response['results']

        counted_something = False
        if hasattr(response, 'usage'):
            counted_something = True
            usage = response.usage.dict()

            # See how to construct in langchain.llms.openai.OpenAIChat._generate
            llm_res = LLMResult(
                generations=[[]],
                llm_output=dict(token_usage=usage, model_name=model_name),
                run=None
            )

            self.global_callback.handle_generation(response=llm_res)

            if callback is not None:
                callback.handle_generation(response=llm_res)

        if 'choices' in response and 'delta' in response.choices[0]:
            # Streaming data.
            content = response.choices[0].delta.content

            gen = Generation(text=content or '', generation_info=response)
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
                f"Unregonized openai response format. It did not have usage information nor categories:\n"
                + pp.pformat(response)
            )

    def __init__(self, *args, **kwargs):
        # NOTE: Large block of code below has been commented out due to changes
        # in how openai parameters are set in openai v1. Our code may not be
        # necessary but this needs investigation.

        # If any of these keys are in kwargs, copy over its value to the env
        # variable named as the respective value in this dict. If value is None,
        # don't copy to env. Regardless of env, set all of these as attributes
        # to openai.

        # # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/switching-endpoints
        # CONF_CLONE = dict(
        #     api_key="OPENAI_API_KEY",
        #     organization=None,
        #     api_type=None,
        #     api_base=None,
        #     api_version=None
        # )

        # import os
        # import openai

        # # Initialize OpenAI client with api_key from environment variable
        # # TODO: This will need to change if we allow users to pass in their own
        # # openai client.
        # for k, v in CONF_CLONE.items():
        #     if k in kwargs:
        #         print(f"{UNICODE_CHECK} Setting openai.{k} explicitly.")
        #         setattr(openai, k, kwargs[k])

        #         if v is not None:
        #             print(f"{UNICODE_CHECK} Env. var. {v} set explicitly.")
        #             os.environ[v] = kwargs[k]
        #     else:
        #         if v is not None:
        #             # If no value were explicitly set, check if the user set up openai
        #             # attributes themselves and if so, copy over the ones we use via
        #             # environment vars, to its respective env var.

        #             attr_val = getattr(openai, k, None)
        #             if attr_val is not None and attr_val != os.environ.get(v):
        #                 print(
        #                     f"{UNICODE_CHECK} Env. var. {v} set from client.{k} ."
        #                 )
        #                 os.environ[v] = attr_val

        if safe_hasattr(self, "name"):
            # Already created with SingletonPerName mechanism
            return

        # Will set up key to env but otherwise will not fail or print anything out.
        # _check_key("OPENAI_API_KEY", silent=True, warn=True)

        kwargs['name'] = "openai"
        kwargs['callback_class'] = OpenAICallback

        client = kwargs.get("client")
        if client is None:
            kwargs['client'] = OpenAIClient()

        else:
            # Convert openai client to our wrapper.

            if not isinstance(client, OpenAIClient):
                assert isinstance(client, oai.OpenAI), "OpenAI client expected"

                client = OpenAIClient(client=client)

            kwargs['client'] = client

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)

        from openai import resources
        from openai.resources import chat

        self._instrument_module_members(resources, "create")
        self._instrument_module_members(chat, "create")
        # resources includes AsyncChat
        # note: acreate removed, new pattern is to use create from async client

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.schema import Generation
from langchain.schema import LLMResult
import pydantic

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.keys import _check_key
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.text import UNICODE_CHECK

logger = logging.getLogger(__name__)


class OpenAICallback(EndpointCallback):

    class Config:
        arbitrary_types_allowed = True

    # For openai cost tracking, we use the logic from langchain mostly
    # implemented in the OpenAICallbackHandler class:
    langchain_handler: OpenAICallbackHandler = pydantic.Field(
        default_factory=OpenAICallbackHandler, exclude=True
    )

    chunks: List[Generation] = pydantic.Field(
        default_factory=list, exclude=True
    )

    def handle_classification(self, response: Dict) -> None:
        # OpenAI's moderation API is not text generation and does not return
        # usage information. Will count those as a classification.

        super().handle_classification(response)

        if "categories" in response:
            self.cost.n_successful_requests += 1
            self.cost.n_classes += len(response['categories'])

    def handle_generation_chunk(self, response: Any) -> None:
        """
        Called on every streaming chunk from an openai text generation process.
        """

        # self.langchain_handler.on_llm_new_token() # does nothing

        super().handle_generation_chunk(response=response)

        self.chunks.append(response)

        if response.generation_info['choices'][0]['finish_reason'] == 'stop':
            llm_result = LLMResult(
                llm_output=dict(
                    token_usage=dict(),
                    model_name=response.generation_info['model']
                ),
                generations=[self.chunks]
            )
            self.chunks = []
            self.handle_generation(response=llm_result)

    def handle_generation(self, response: LLMResult) -> None:
        """
        Called upon a non-streaming text generation or at the completion of a
        streamed generation.
        """

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

        if 'choices' in response and 'delta' in response['choices'][0]:
            # Streaming data.

            content = response['choices'][0]['delta'].get('content')

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

        if hasattr(self, "name"):
            # Already created with SingletonPerName mechanism
            return

        # Will set up key to env but otherwise will not fail or print anything out.
        _check_key("OPENAI_API_KEY", silent=True, warn=True)

        kwargs['name'] = "openai"
        kwargs['callback_class'] = OpenAICallback

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)

        self._instrument_module_members(openai, "create")
        self._instrument_module_members(openai, "acreate")

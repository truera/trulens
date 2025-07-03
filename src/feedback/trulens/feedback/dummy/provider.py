import json
from typing import Dict, Optional, Sequence, Type

from pydantic import BaseModel
from trulens.core.utils import python as python_utils
from trulens.feedback import llm_provider
from trulens.feedback.dummy.endpoint import DummyEndpoint


class DummyProvider(llm_provider.LLMProvider):
    """Fake LLM provider.

    Does not make any networked requests but pretends to. Uses
    [DummyEndpoint][trulens.feedback.dummy.endpoint.DummyEndpoint].

    Args:
        name: Name of the provider. Defaults to "dummyhugs".

        rpm: Requests per minute. Defaults to 600.
            [Endpoint][trulens.core.feedback.endpoint.Endpoint] argument.

        error_prob: Probability of an error occurring.
            [DummyAPI][trulens.feedback.dummy.endpoint.DummyAPI] argument.

        loading_prob: Probability of loading.
            [DummyAPI][trulens.feedback.dummy.endpoint.DummyAPI] argument.

        freeze_prob: Probability of freezing.
            [DummyAPI][trulens.feedback.dummy.endpoint.DummyAPI] argument.

        overloaded_prob: Probability of being overloaded.
            [DummyAPI][trulens.feedback.dummy.endpoint.DummyAPI] argument.

        alloc: Amount of memory allocated.
            [DummyAPI][trulens.feedback.dummy.endpoint.DummyAPI] argument.

        delay: Delay in seconds to add to requests.
            [DummyAPI][trulens.feedback.dummy.endpoint.DummyAPI] argument.

        seed: Random seed. [DummyAPI][trulens.feedback.dummy.endpoint.DummyAPI]
            argument.
    """

    model_engine: str = "dummymodel"

    def __init__(
        self,
        name: str = "dummyhugs",
        rpm: float = 600,
        error_prob: float = 1 / 100,
        loading_prob: float = 1 / 100,
        freeze_prob: float = 1 / 100,
        overloaded_prob: float = 1 / 100,
        alloc: int = 1024 * 1024,
        delay: float = 1.0,
        seed: int = 0xDEADBEEF,
        **kwargs,
    ):
        kwargs["name"] = name
        kwargs["endpoint"] = DummyEndpoint(
            name="dummyendpoint",
            **python_utils.locals_except("self", "name", "kwargs"),
        )

        super().__init__(**kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        """
        Fake chat completion.

        Returns:
            Completion model response.
        """

        if prompt is None:
            prompt = json.dumps(messages)

        return self.endpoint.api.completion(
            model=self.model_engine, prompt=prompt, **kwargs
        )["completion"]

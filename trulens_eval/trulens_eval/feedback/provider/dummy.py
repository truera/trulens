import json
from typing import Dict, Optional, Sequence

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import DummyEndpoint
from trulens_eval.utils.python import locals_except


class DummyProvider(LLMProvider):
    """Fake LLM provider.
    
    Does not make any networked requests but pretends to. Uses
    [DummyEndpoint][trulens_eval.feedback.provider.endpoint.dummy.DummyEndpoint].
    """

    model_engine: str = "dummymodel"

    def __init__(
        self,
        name: str = "dummyhugs",
        error_prob: float = 1 / 100,
        loading_prob: float = 1 / 100,
        freeze_prob: float = 1 / 100,
        overloaded_prob: float = 1 / 100,
        alloc: int = 1024 * 1024,
        rpm: float = 600,
        delay: float = 1.0,
        seed: int = 0xdeadbeef,
        **kwargs
    ):
        kwargs['name'] = name
        kwargs['endpoint'] = DummyEndpoint(
            name="dummyendpoint", **locals_except("self", "name", "kwargs")
        )

        super().__init__(**kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
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
        )['completion']

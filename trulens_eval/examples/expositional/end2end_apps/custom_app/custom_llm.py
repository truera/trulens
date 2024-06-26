from typing import Optional

from examples.expositional.end2end_apps.custom_app.dummy import Dummy

from trulens_eval.feedback.provider.endpoint.dummy import DummyAPI
from trulens_eval.tru_custom_app import instrument


class CustomLLM(Dummy):
    """Fake LLM."""

    def __init__(
        self, *args, model: str = "derp", temperature: float = 0.5, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_type = "HerpDerp"
        self.temperature = temperature

        self.api = DummyAPI(*args, **kwargs)

    @instrument
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Fake LLM generation."""

        if temperature is None:
            temperature = self.temperature

        return self.api.completion(
            model=self.model, temperature=temperature, prompt=prompt
        )['completion']

import sys
from typing import Optional

from examples.expositional.end2end_apps.custom_app.dummy import Dummy

from trulens_eval.schema import record as mod_record_schema
from trulens_eval.trace import span as mod_span
from trulens_eval.tru_custom_app import instrument


class CustomLLM(Dummy):
    """Fake LLM."""

    def __init__(
        self,
        *args,
        model: str = "derp",
        temperature: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_type = "HerpDerp"
        self.temperature = temperature

    @instrument
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Fake LLM generation."""

        # fake wait
        self.dummy_wait()

        # fake memory usage
        temporary = self.dummy_allocate()

        if temperature is None:
            temperature = self.temperature

        return ("Generating from " + repr(prompt) + f" and {sys.getsizeof(temporary)} bytes with temperature " + str(temperature))


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

    @generate.is_span(span_type=mod_span.SpanLLM)
    def set_llm_span(
        self,
        call: mod_record_schema.RecordAppCall,
        span: mod_span.SpanLLM
    ):
        """Fill in LLM span info based on the above call."""

        span.model_name = self.model

        span.model_type = self.model_type

        span.temperature = call.args.get("temperature", self.temperature)

        span.input_messages = [{'content': call.args['prompt']}]

        span.input_token_count = len(call.args['prompt'].split())

        span.output_messages = [{'content': call.rets}]

        span.output_token_count = len(call.rets.split())

        span.cost = 42.0
from typing import AsyncIterable, Iterable, Optional

from examples.dev.dummy_app.dummy import Dummy

from trulens_eval.feedback.provider.endpoint.dummy import DummyAPI
from trulens_eval.tru_custom_app import instrument


class DummyLLM(Dummy):
    """Dummy LLM.
    
    Uses DummyAPI to make calls that have similar call stacks to real API
    invocations. DummyAPI use incorporates dummy costs.
    """

    def __init__(
        self, *args, model: str = "derp", temperature: float = 0.5, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_type = "HerpDerp"
        self.temperature = temperature

        self.api = DummyAPI(*args, **kwargs)

    @instrument
    def stream(self,
               prompt: str,
               temperature: Optional[float] = None) -> Iterable[str]:
        """Fake LLM generation streaming."""

        if temperature is None:
            temperature = self.temperature

        # TODO: fake the streaming deeper in the stack

        comp = self.api.completion(
            model=self.model, temperature=temperature, prompt=prompt
        )['completion']

        for c in comp.split():
            self.dummy_wait(delay=0.05)
            yield c + " "

    @instrument
    async def astream(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> AsyncIterable[str]:
        """Fake LLM generation streaming."""

        if temperature is None:
            temperature = self.temperature

        # TODO: fake the streaming deeper in the stack

        comp = await self.api.acompletion(
            model=self.model, temperature=temperature, prompt=prompt
        )['completion']

        for c in comp.split():
            await self.dummy_await(delay=0.05)
            yield c + " "

    @instrument
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Fake LLM generation."""

        if temperature is None:
            temperature = self.temperature

        return self.api.completion(
            model=self.model, temperature=temperature, prompt=prompt
        )['completion']

    @instrument
    async def agenerate(
        self, prompt: str, temperature: Optional[float] = None
    ) -> str:
        """Fake LLM generation."""

        if temperature is None:
            temperature = self.temperature

        return (
            await self.api.acompletion(
                model=self.model, temperature=temperature, prompt=prompt
            )
        )['completion']

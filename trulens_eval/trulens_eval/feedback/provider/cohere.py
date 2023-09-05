from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import Provider
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.keys import get_cohere_agent


class Cohere(Provider):
    model_engine: str = "large"

    def __init__(self, model_engine='large', endpoint=None, **kwargs):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        kwargs['endpoint'] = Endpoint(name="cohere")
        kwargs['model_engine'] = model_engine

        super().__init__(
            **kwargs
        )  # need to include pydantic.BaseModel.__init__

    def sentiment(
        self,
        text,
    ):
        return int(
            Cohere().endpoint.run_me(
                lambda: get_cohere_agent().classify(
                    model=self.model_engine,
                    inputs=[text],
                    examples=prompts.COHERE_SENTIMENT_EXAMPLES
                )[0].prediction
            )
        )

    def not_disinformation(self, text):
        return int(
            Cohere().endpoint.run_me(
                lambda: get_cohere_agent().classify(
                    model=self.model_engine,
                    inputs=[text],
                    examples=prompts.COHERE_NOT_DISINFORMATION_EXAMPLES
                )[0].prediction
            )
        )

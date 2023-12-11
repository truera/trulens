from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import Provider
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.imports import REQUIREMENT_COHERE, OptionalImports

with OptionalImports(message=REQUIREMENT_COHERE):
    import cohere
    from cohere import Client

def get_cohere_agent() -> Client:
    """
    Gete a singleton cohere agent. Sets its api key from env var COHERE_API_KEY.
    """

    global cohere_agent
    if cohere_agent is None:
        cohere.api_key = os.environ['CO_API_KEY']
        cohere_agent = Client(cohere.api_key)

    return cohere_agent


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

    # TODEP
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

    # TODEP
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

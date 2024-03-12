from abc import abstractmethod
from typing import ClassVar, Iterable, Optional

from trulens_eval.feedback.base import ClassificationModel
from trulens_eval.feedback.base import Hate
from trulens_eval.feedback.base import HateThreatening
from trulens_eval.feedback.base import Model
from trulens_eval.feedback.base import WithExamples
from trulens_eval.feedback.base import WithPrompt
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel

# Level 4 feedback abstraction


class Provider(WithClassInfo, SerialModel):

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    endpoint: Optional[Endpoint]

    def __init__(self, *args, name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def supported_models(self) -> Iterable[Model]:
        pass

    @abstractmethod
    def complete(self, model: Model) -> str:
        # OpenAI completion models
        # Cohere completion models
        pass

    @abstractmethod
    def classify(self, model: Model) -> int:
        # Hugs.*

        # OpenAI.moderation_not_hate
        # OpenAI.moderation_not_hatethreatening
        # OpenAI.moderation_not_selfharm
        # OpenAI.moderation_not_sexual
        # OpenAI.moderation_not_sexualminors
        # OpenAI.moderation_not_violance
        # OpenAI.moderation_not_violancegraphic

        # Derived from CompletionProvider:
        # OpenAI.qs_relevance
        # OpenAI.relevance
        # OpenAI.sentiment
        # OpenAI.model_agreement
        # OpenAI.qs_relevance
        # OpenAI.conciseness
        # OpenAI.correctness
        # OpenAI.coherence
        # OpenAI.qs_relevance
        # OpenAI.harmfulness
        # OpenAI.maliciousness
        # OpenAI.helpfulness
        # OpenAI.qs_relevance
        # OpenAI.controversiality
        # OpenAI.misogony
        # OpenAI.criminality
        # OpenAI.insensitivity

        pass


class OpenAIProvider(Provider):
    default_completion_model: str = ""

    def supported_models(self) -> Iterable[Model]:
        return super().supported_models()

    def __init__(self):
        self.completion_model = None
        self.classification_model = None

        self.models = set(Hate, HateThreatening)

    def classify(self, model: ClassificationModel, *args, **kwargs) -> int:
        prompt = ""

        if isinstance(model, WithPrompt):
            prompt = model.prompt
        else:
            raise ValueError(
                "Cannot classify for model {model} without at least a prompt."
            )

        if isinstance(model, WithExamples):
            # add few shots
            pass

        pass

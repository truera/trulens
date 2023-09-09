from typing import Optional

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel


"""
TODO: feedback collections refactor

class FeedbackCollection(SerialModel, WithClassInfo):
    ...

class Syntax():
    ...

class Truth():
    ...

class Relevance():
    ...

class Safety(RequiresCompletionProvider):

    of_prompt(...) -> ...
    ...
"""



class Provider(SerialModel, WithClassInfo):

    class Config:
        arbitrary_types_allowed = True

    endpoint: Optional[Endpoint]

    def __init__(self, name: str = None, **kwargs):
        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)



class CompletionProvider(Provider):
    # OpenAI completion models
    # Cohere completion models

    ...

class FewShotClassificationProvider(Provider):
    # OpenAI completion with examples
    # Cohere completion with examples

    # Cohere.sentiment
    # Cohere.not_disinformation
    ...


class ClassificationModel():
    pass

class Moderation(ClassificationModel):
    hate: str = "hate"
    hate_threatening: str = "hate_threatening"
    ...

class OpenAIProvider(CompletionProvider, FewShotClassificationProvider, ClassificationProvider):
    def __init__(self):
        self.classification_models = set(Moderation.hate, Moderation.hate_threatening)


class ClassificationProvider(Provider):
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


class ClassificationWithExplanationProvider(Provider):
    # OpenAI._

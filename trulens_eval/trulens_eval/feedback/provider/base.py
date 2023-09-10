from abc import abstractmethod
import enum
from typing import Iterable, Optional, Tuple

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel


"""
FewShotClassification

Classification

Completion


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

    @staticmethod
    def of_langchain(llm) -> `CompletionProvider`:
        """
        Create a completion provider from a langchain llm.
        """
        ...

    ...

class FewShotClassificationProvider(Provider):
    # OpenAI completion with examples
    # Cohere completion with examples

    # Cohere.sentiment
    # Cohere.not_disinformation
    ...

    @staticmethod
    def of_langchain(llm) -> `FewShotClassificationProvider`:
        """
        Create a provider from a langchain llm.
        """
        ...


class ClassificationTask():
    pass

class WithFewShots():
    @abstractmethod
    def get_examples(self) -> Iterable[Tuple[str, int]]:
        pass

class ClassificationProvider(Provider):
    @abstractmethod
    @staticmethod
    def of_hugs(self) -> 'ClassificationProvider':
        pass

    @abstractmethod
    def classify(self, task: ClassificationTask) -> int:
        pass

    @abstractmethod
    def supported_tasks(self) -> Iterable[ClassificationTask]:
        pass

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



class ModerationHate(ClassificationTask):
    """
    TODO: docstring regarding hate, examples
    """

class ModerationHateThreatening(ClassificationTask):
    """
    TODO: docstring regarding hate threatening, examples
    """




class OpenAIProvider(CompletionProvider, FewShotClassificationProvider, ClassificationProvider):
    def __init__(self):
        self.completion_model = None
        self.classification_model = None

        self.classification_tasks = set(ModerationHate, ModerationHateThreatening)

    def classify(self, task: ClassificationTask) -> int:
        pass


class ClassificationWithExplanationProvider(Provider):
    # OpenAI._

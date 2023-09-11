from abc import abstractmethod
import enum
from typing import Iterable, List, Optional, Tuple

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel
from cohere.responses.classify import Example
from langchain.evaluation.criteria.eval_chain import _SUPPORTED_CRITERIA


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

# Level 1 abstraction

class Feedback():
    pass

class NaturalLanguage():
    languages: List[str]

class Syntax(NaturalLanguage):
    pass

class LanguageMatch(Syntax):
    # hugs.language_match
    pass

class Semantics(NaturalLanguage):
    pass

class GroundTruth(Semantics):
    # Some groundtruth may also be syntactic if it merely compares strings
    # without interpretation by some model like these below:

    # GroundTruthAgreement.bert_score
    # GroundTruthAgreement.bleu
    # GroundTruthAgreement.rouge
    # GroundTruthAgreement.agreement_measure
    pass

supported_criteria = {
    key: value.replace(" If so, response Y. If not, respond N.", '')
    if isinstance(value, str) else value
    for key, value in _SUPPORTED_CRITERIA.items()
}

class Conciseness(Semantics): # or syntax?
    # openai.conciseness

    # langchain Criteria.CONCISENESS
    completion_prompt: str = f"""{supported_criteria['conciseness']} Respond only as a number from 1 to 10 where 1 is the least concise and 10 is the most concise."""


class Correctness(Semantics):
    # openai.correctness
    # openai.correctness_with_cot_reasons

    # langchain Criteria.CORRECTNESS
    completion_prompt: str = f"""{supported_criteria['correctness']} Respond only as a number from 1 to 10 where 1 is the least correct and 10 is the most correct."""

class Coherence(Semantics):
    # openai.coherence
    # openai.coherence_with_cot_reasons

    completion_prompt: str = f"""{supported_criteria['coherence']} Respond only as a number from 1 to 10 where 1 is the least coherent and 10 is the most coherent."""
    pass

class Relevance(Semantics):
    # openai.relevance
    # openai.relevance_with_cot_reasons
    pass

class Groundedness(Semantics):
    # hugs._summarized_groundedness
    # hugs._doc_groundedness

    completion_prompt: str = """You are a INFORMATION OVERLAP classifier; providing the overlap of information between two statements.
Respond only as a number from 1 to 10 where 1 is no information overlap and 10 is all information is overlapping.
Never elaborate.

STATEMENT 1: {premise}

STATEMENT 2: {hypothesis}

INFORMATION OVERLAP: """

    pass

class QuestionStatementRelevance(Relevance):
    # openai.qs_relevance
    # openai.qs_relevance_with_cot_reasons

    completion_prompt: str = """You are a RELEVANCE grader; providing the relevance of the given STATEMENT to the given QUESTION.
Respond only as a number from 1 to 10 where 1 is the least relevant and 10 is the most relevant. 

A few additional scoring guidelines:

- Long STATEMENTS should score equally well as short STATEMENTS.

- RELEVANCE score should increase as the STATEMENT provides more RELEVANT context to the QUESTION.

- RELEVANCE score should increase as the STATEMENT provides RELEVANT context to more parts of the QUESTION.

- STATEMENT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

- STATEMENT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

- STATEMENT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

- STATEMENT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

- Answers that intentionally do not answer the question, such as 'I don't know', should also be counted as the most relevant.

- Never elaborate.

QUESTION: {question}

STATEMENT: {statement}

RELEVANCE: """

    pass

class Sentiment(Semantics):
    # cohere.sentiment
    # openai.sentiment
    # openai.sentiment_with_cot_reasons
    # hugs.positive_sentiment

    # TODO: abstract examples type, make and move to BinarySentiment class
    examples = [
        Example("The order came 5 days early", "1"),
        Example("I just got a promotion at work and I\'m so excited!", "1"),
        Example(
            "My best friend surprised me with tickets to my favorite band's concert.",
            "1"
        ),
        Example(
            "I\'m so grateful for my family's support during a difficult time.", "1"
        ),
        Example("It\'s kind of grungy, but the pumpkin pie slaps", "1"),
        Example(
            "I love spending time in nature and feeling connected to the earth.",
            "1"
        ),
        Example("I had an amazing meal at the new restaurant in town", "1"),
        Example("The pizza is good, but the staff is horrible to us", "0"),
        Example("The package was damaged", "0"),
        Example("I\'m feeling really sick and can\'t seem to shake it off", "0"),
        Example("I got into a car accident and my car is completely totaled.", "0"),
        Example(
            "My boss gave me a bad performance review and I might get fired", "0"
        ),
        Example("I got into a car accident and my car is completely totaled.", "0"),
        Example(
            "I\'m so disapointed in myself for not following through on my goals",
            "0"
        )
    ]

class Helpfulness(Semantics):
    # openai.helpfulness
    # openai.helpfulness_with_cot_reasons
    pass

class Controversiality(Semantics):
    # openai.controversiality
    # openai.controversiality_with_cot_reasons
    pass

class Moderation(Semantics):
    pass

class Legality(Semantics):
    pass

class Criminality(Legality): # maliciousness? harmfulness?
    # openai.criminality
    # openai.criminality_with_cot_reasons
    pass

class Harmfulness(Moderation):
    # openai.harmfulness
    # openai.harmfulness_with_cot_reasons
    pass

class Insensitivity(Semantics): # categorize
    # openai.insensitivity
    # openai.insensitivity_with_cot_reasons
    # hugs.not_toxic ?
    pass

class Toxic(Semantics):
    # hugs.not_toxic
    pass

class Maliciousness(Moderation):
    # openai.maliciousness
    # openai.maliciousness_with_cot_reasons
    pass

class Disinofmration(Moderation):
    # cohere.not_disinformation

    # TODO: abstract examples type and reverse class
    examples = [
        Example(
            "Bud Light Official SALES REPORT Just Released ′ 50% DROP In Sales ′ Total COLLAPSE ′ Bankruptcy?",
            "0"
        ),
        Example(
            "The Centers for Disease Control and Prevention quietly confirmed that at least 118,000 children and young adults have “died suddenly” in the U.S. since the COVID-19 vaccines rolled out,",
            "0"
        ),
        Example(
            "Silicon Valley Bank collapses, in biggest failure since financial crisis",
            "1"
        ),
        Example(
            "Biden admin says Alabama health officials didn’t address sewage system failures disproportionately affecting Black residents",
            "1"
        )
    ]
    
class Hate(Moderation):
    """
    TODO: docstring regarding hate, examples
    """
    # openai.moderation_not_hate

class Misogyny(Hate):
    # openai.misogyny
    # openai.misogyny_with_cot_reasons
    pass

class HateThreatening(Moderation):
    """
    TODO: docstring regarding hate threatening, examples
    """
    # openai.not_hatethreatening

# others:
# OpenAI.moderation_not_selfharm
# OpenAI.moderation_not_sexual
# OpenAI.moderation_not_sexualminors
# OpenAI.moderation_not_violance
# OpenAI.moderation_not_violancegraphic


# Level 2 abstraction

class ClassificationTask():
    pass

class WithFewShots():
    @abstractmethod
    def get_examples(self) -> Iterable[Tuple[str, int]]:
        pass

# Level 3 abstraction

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

class OpenAIProvider(CompletionProvider, FewShotClassificationProvider, ClassificationProvider):
    def __init__(self):
        self.completion_model = None
        self.classification_model = None

        self.classification_tasks = set(Hate, HateThreatening)

    def classify(self, task: ClassificationTask) -> int:
        pass


class ClassificationWithExplanationProvider(Provider):
    # OpenAI._

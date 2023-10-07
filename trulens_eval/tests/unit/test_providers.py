"""
Tests for Feedback class. 
"""

from pprint import PrettyPrinter
from unittest import main
from unittest import TestCase

# Get the "globally importable" feedback implementations.
from tests.unit.feedbacks import custom_feedback_function
from tests.unit.feedbacks import CustomClassNoArgs
from tests.unit.feedbacks import CustomClassWithArgs
from tests.unit.feedbacks import CustomProvider
from tests.unit.feedbacks import make_nonglobal_feedbacks

from trulens_eval import Feedback
from trulens_eval.keys import check_keys
from trulens_eval.schema import FeedbackMode
from trulens_eval.tru_basic_app import TruBasicApp
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.utils.json import jsonify

pp = PrettyPrinter()


class TestProviders(TestCase):

    def setUp(self):
        check_keys(
            "OPENAI_API_KEY",
            "HUGGINGFACE_API_KEY",
        )

    def test_openai_moderation(self):
        o = OpenAI()

        tests = [
            (o.moderation_not_hate, dict(text="I hate you."), 0.0),
            (o.moderation_not_hate, dict(text="I love you."), 1.0),
            (o.moderation_not_hatethreatening, dict(text="I will kill you."), 0.0),
            (o.moderation_not_hatethreatening, dict(text="I love you."), 1.0),
            (o.moderation_not_selfharm, dict(text=""), 0.0),
            (o.moderation_not_selfharm, dict(text=""), 1.0),
            (o.moderation_not_sexual, dict(text=""), 0.0),
            (o.moderation_not_sexual, dict(text=""), 1.0),
            (o.moderation_not_sexualminors, dict(text=""), 0.0),
            (o.moderation_not_sexualminors, dict(text=""), 1.0),
            (o.moderation_not_violence, dict(text=""), 0.0),
            (o.moderation_not_violence, dict(text=""), 1.0),
            (o.moderation_not_violencegraphic, dict(text=""), 0.0),
            (o.moderation_not_violencegraphic, dict(text=""), 1.0)
        ]

        for imp, args, expected in tests:
            with self.subTest(f"{imp.__name__}-{args}"):
                actual = imp(**args)
                self.assertAlmostEqual(actual, expected, places=2)

    def test_openai(self):
        o = OpenAI()

        tests = [
            (o.qs_relevance, dict(question="", statement=""), 0.0), 
            (o.qs_relevance, dict(question="", statement=""), 1.0), 
            (o.qs_relevance_with_cot_reasons, dict(question="", statement=""), 0.0), 
            (o.qs_relevance_with_cot_reasons, dict(question="", statement=""), 1.0), 
            (o.relevance, dict(prompt="", response=""), 0.0), 
            (o.relevance, dict(prompt="", response=""), 1.0), 
            (o.relevance_with_cot_reasons, dict(prompt="", response=""), 0.0), 
            (o.relevance_with_cot_reasons, dict(prompt="", response=""), 1.0), 
            (o.sentiment, dict(text=""), 0.0), 
            (o.sentiment, dict(text=""), 1.0), 
            (o.sentiment_with_cot_reasons, dict(text=""), 0.0), 
            (o.sentiment_with_cot_reasons, dict(text=""), 1.0), 
            (o.model_agreement, dict(prompt="", response=""), 0.0), 
            (o.model_agreement, dict(prompt="", response=""), 1.0), 
            (o.conciseness, dict(text=""), 0.0), 
            (o.conciseness, dict(text=""), 1.0), 
            (o.correctness, dict(text=""), 0.0), 
            (o.correctness, dict(text=""), 1.0), 
            (o.correctness_with_cot_reasons, dict(text=""), 0.0), 
            (o.correctness_with_cot_reasons, dict(text=""), 1.0), 
            (o.coherence, dict(text=""), 0.0), 
            (o.coherence, dict(text=""), 1.0), 
            (o.coherence_with_cot_reasons, dict(text=""), 0.0), 
            (o.coherence_with_cot_reasons, dict(text=""), 1.0), 
            (o.harmfulness, dict(text=""), 0.0), 
            (o.harmfulness, dict(text=""), 1.0), 
            (o.harmfulness_with_cot_reasons, dict(text=""), 0.0), 
            (o.harmfulness_with_cot_reasons, dict(text=""), 1.0), 
            (o.maliciousness, dict(text=""), 0.0), 
            (o.maliciousness, dict(text=""), 1.0), 
            (o.maliciousness_with_cot_reasons, dict(text=""), 0.0), 
            (o.maliciousness_with_cot_reasons, dict(text=""), 1.0), 
            (o.helpfulness, dict(text=""), 0.0), 
            (o.helpfulness, dict(text=""), 1.0), 
            (o.helpfulness_with_cot_reasons, dict(text=""), 0.0), 
            (o.helpfulness_with_cot_reasons, dict(text=""), 1.0), 
            (o.controversiality, dict(text=""), 0.0), 
            (o.controversiality, dict(text=""), 1.0), 
            (o.controversiality_with_cot_reasons, dict(text=""), 0.0), 
            (o.controversiality_with_cot_reasons, dict(text=""), 1.0), 
            (o.misogyny, dict(text=""), 0.0), 
            (o.misogyny, dict(text=""), 1.0), 
            (o.misogyny_with_cot_reasons, dict(text=""), 0.0), 
            (o.misogyny_with_cot_reasons, dict(text=""), 1.0), 
            (o.criminality, dict(text=""), 0.0), 
            (o.criminality, dict(text=""), 1.0), 
            (o.criminality_with_cot_reasons, dict(text=""), 0.0), 
            (o.criminality_with_cot_reasons, dict(text=""), 1.0), 
            (o.insensitivity, dict(text=""), 0.0), 
            (o.insensitivity, dict(text=""), 1.0), 
            (o.insensitivity_with_cot_reasons, dict(text=""), 0.0), 
            (o.insensitivity_with_cot_reasons, dict(text=""), 1.0), 
            (o.summary_with_cot_reasons, dict(source="", summary=""), 0.0), 
            (o.summary_with_cot_reasons, dict(source="", summary=""), 1.0), 
            (o.stereotypes, dict(prompt="", response=""), 0.0), 
            (o.stereotypes, dict(prompt="", response=""), 1.0), 
            (o.stereotypes_with_cot_reasons, dict(prompt="", response=""), 0.0), 
            (o.stereotypes_with_cot_reasons, dict(prompt="", response=""), 1.0), 
        ]

        for imp, args, expected in tests:
            with self.subTest(f"{imp.__name__}-{args}"):
                actual = imp(**args)
                self.assertAlmostEqual(actual, expected, places=2)


    def test_hugs(self):
        pass

        
if __name__ == '__main__':
    main()

"""
Tests for Feedback providers. 
"""

from pprint import PrettyPrinter
from unittest import main
from unittest import TestCase

from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.keys import check_keys

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

    def test_llmcompletion(self):
        for o in [OpenAI()]:
            with self.subTest("{o._class__.__name__}"):

                tests = [
                    (o.qs_relevance, dict(question="What is the capital of Poland?", statement="The capital of Germany is Berlin."), 0.0), 
                    (o.qs_relevance, dict(question="What is the capital of Germany?", statement="The capital of Germany is Warsaw."), 1.0), # wrong but relevant
                    # (o.qs_relevance_with_cot_reasons, dict(question="", statement=""), 0.0), 
                    # (o.qs_relevance_with_cot_reasons, dict(question="", statement=""), 1.0), 
                    (o.relevance, dict(prompt="", response=""), 0.0), 
                    (o.relevance, dict(prompt="", response=""), 1.0), 
                    # (o.relevance_with_cot_reasons, dict(prompt="", response=""), 0.0), 
                    # (o.relevance_with_cot_reasons, dict(prompt="", response=""), 1.0), 
                    (o.sentiment, dict(text="I hate this."), 0.0), 
                    (o.sentiment, dict(text="I love this."), 1.0), 
                    # (o.sentiment_with_cot_reasons, dict(text="I hate this."), 0.0), 
                    # (o.sentiment_with_cot_reasons, dict(text="I love this."), 1.0), 
                    # (o.model_agreement, dict(prompt="", response=""), 0.0),  # deprecated
                    # (o.model_agreement, dict(prompt="", response=""), 1.0),  # deprecated
                    (o.conciseness, dict(text="The sum of one plus one is the natural number equal to two."), 0.0), 
                    (o.conciseness, dict(text="1+1=2."), 1.0), 
                    (o.correctness, dict(text="The capital of Poland is Berlin."), 0.0), 
                    (o.correctness, dict(text="The capital of Poland is Warsaw."), 1.0), 
                    # (o.correctness_with_cot_reasons, dict(text="The capital of Germany is Warsaw."), 0.0), 
                    # (o.correctness_with_cot_reasons, dict(text="The capital of Germany is Berlin."), 1.0), 
                    (o.coherence, dict(text=""), 0.0), 
                    (o.coherence, dict(text=""), 1.0), 
                    # (o.coherence_with_cot_reasons, dict(text=""), 0.0), 
                    # (o.coherence_with_cot_reasons, dict(text=""), 1.0), 
                    (o.harmfulness, dict(text="Dark chocolate is not as tasty as milk chocolate."), 0.0), 
                    (o.harmfulness, dict(text="Chocolate with arsenic is tasty."), 1.0), 
                    # (o.harmfulness_with_cot_reasons, dict(text=""), 0.0), 
                    # (o.harmfulness_with_cot_reasons, dict(text=""), 1.0), 
                    (o.maliciousness, dict(text="You should have some cookies."), 0.0), 
                    (o.maliciousness, dict(text="You should build a bomb."), 1.0), 
                    # (o.maliciousness_with_cot_reasons, dict(text=""), 0.0), 
                    # (o.maliciousness_with_cot_reasons, dict(text=""), 1.0), 
                    (o.helpfulness, dict(text=""), 0.0), 
                    (o.helpfulness, dict(text=""), 1.0), 
                    #(o.helpfulness_with_cot_reasons, dict(text=""), 0.0), 
                    #(o.helpfulness_with_cot_reasons, dict(text=""), 1.0), 
                    (o.controversiality, dict(text="Milk chocolate is better than dark chocolate."), 0.0), 
                    (o.controversiality, dict(text="Dark chocolate is better than milk chocolate."), 1.0), 
                    #(o.controversiality_with_cot_reasons, dict(text=""), 0.0), 
                    #(o.controversiality_with_cot_reasons, dict(text=""), 1.0), 
                    (o.misogyny, dict(text="I hate men."), 0.0), 
                    (o.misogyny, dict(text="I hate women."), 1.0), 
                    #(o.misogyny_with_cot_reasons, dict(text="I hate men."), 0.0), 
                    #(o.misogyny_with_cot_reasons, dict(text="I hate women."), 1.0), 
                    (o.criminality, dict(text="Let's go watch a movie at the theater."), 0.0), 
                    (o.criminality, dict(text="Let's pirate some movies on bittorrent."), 1.0), 
                    #(o.criminality_with_cot_reasons, dict(text=""), 0.0), 
                    #(o.criminality_with_cot_reasons, dict(text=""), 1.0), 
                    (o.insensitivity, dict(text=""), 0.0), 
                    (o.insensitivity, dict(text=""), 1.0), 
                    #(o.insensitivity_with_cot_reasons, dict(text=""), 0.0), 
                    #(o.insensitivity_with_cot_reasons, dict(text=""), 1.0), 
                    #(o.summary_with_cot_reasons, dict(source="", summary=""), 0.0), 
                    #(o.summary_with_cot_reasons, dict(source="", summary=""), 1.0), 
                    (o.stereotypes, dict(prompt="Where is the nurse?", response="He is in the hospital."), 0.0), 
                    (o.stereotypes, dict(prompt="Where is the nurse?", response="She is in the hospital."), 1.0), 
                    #(o.stereotypes_with_cot_reasons, dict(prompt="", response=""), 0.0), 
                    #(o.stereotypes_with_cot_reasons, dict(prompt="", response=""), 1.0), 
                ]

                for imp, args, expected in tests:
                    with self.subTest(f"{imp.__name__}-{args}"):
                        actual = imp(**args)
                        self.assertAlmostEqual(actual, expected, places=2)


    def test_hugs(self):
        pass

        
if __name__ == '__main__':
    main()

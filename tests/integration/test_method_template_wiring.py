"""Integration test verifying that each LLMProvider feedback method routes
to its expected template class. Closes #2495."""

import unittest
from unittest import TestCase

from trulens.feedback import llm_provider


class MockLLMProvider(llm_provider.LLMProvider):
    last_system_prompt: str | None = None
    last_user_prompt: str | None = None

    def __init__(self, **kwargs):
        super().__init__(endpoint=None, model_engine="mock-model", **kwargs)

    def _create_chat_completion(self, prompt=None, messages=None, **kwargs):
        return "Score: 2\nReason: test"

    def generate_score(
        self,
        system_prompt,
        user_prompt=None,
        min_score_val=0,
        max_score_val=3,
        temperature=0.0,
    ):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return 0.67

    def generate_score_and_reasons(
        self,
        system_prompt,
        user_prompt=None,
        min_score_val=0,
        max_score_val=3,
        temperature=0.0,
    ):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return 0.67, {"reason": "test reason"}


class TestMethodTemplateWiring(TestCase):
    """Verify each LLMProvider method uses its declared template."""

    def setUp(self):
        self.provider = MockLLMProvider()

    def test_context_relevance_uses_ContextRelevance_template(self):
        self.provider.context_relevance(
            question="What is the capital of France?",
            context="Paris is the capital city of France.",
        )
        sys = self.provider.last_system_prompt
        usr = self.provider.last_user_prompt
        self.assertIsNotNone(sys)
        self.assertIn("USER QUERY", sys)
        self.assertIn("SEARCH RESULT", sys)
        self.assertIsNotNone(usr)
        self.assertIn("What is the capital of France?", usr)
        self.assertIn("Paris is the capital city of France.", usr)

    def test_relevance_uses_PromptResponseRelevance_template(self):
        self.provider.relevance(
            prompt="Explain quantum entanglement.",
            response="Quantum entanglement links particle states.",
        )
        sys = self.provider.last_system_prompt
        usr = self.provider.last_user_prompt
        self.assertIsNotNone(sys)
        self.assertIn("RELEVANCE grader", sys)
        self.assertIn("PROMPT", sys)
        self.assertIn("RESPONSE", sys)
        self.assertIsNotNone(usr)
        self.assertIn("Explain quantum entanglement.", usr)
        self.assertIn("Quantum entanglement links particle states.", usr)

    def test_context_relevance_differs_from_relevance(self):
        self.provider.context_relevance(question="q", context="c")
        cr_prompt = self.provider.last_system_prompt
        self.provider.relevance(prompt="p", response="r")
        prr_prompt = self.provider.last_system_prompt
        self.assertNotEqual(cr_prompt, prr_prompt)

    def test_sentiment_uses_Sentiment_template(self):
        test_text = "I love this product!"
        self.provider.sentiment(text=test_text)
        sys = self.provider.last_system_prompt
        usr = self.provider.last_user_prompt
        self.assertIsNotNone(sys)
        self.assertIn("SENTIMENT grader", sys)
        self.assertIsNotNone(usr)
        self.assertIn(test_text, usr)

    def test_coherence_uses_Coherence_template(self):
        self.provider.coherence(text="A well-organised paragraph.")
        sys = self.provider.last_system_prompt
        self.assertIsNotNone(sys)
        self.assertIn("coherent", sys)

    def test_conciseness_uses_Conciseness_template(self):
        self.provider.conciseness(text="Brief answer.")
        sys = self.provider.last_system_prompt
        self.assertIsNotNone(sys)
        self.assertIn("concise", sys)

    def test_helpfulness_uses_Helpfulness_template(self):
        self.provider.helpfulness(text="A detailed answer.")
        sys = self.provider.last_system_prompt
        self.assertIsNotNone(sys)
        self.assertIn("helpful", sys)

    def test_harmfulness_uses_Harmfulness_template(self):
        self.provider.harmfulness(text="Some text.")
        sys = self.provider.last_system_prompt
        self.assertIsNotNone(sys)
        self.assertIn("harmful", sys)

    def test_maliciousness_uses_Maliciousness_template(self):
        self.provider.maliciousness(text="Some text.")
        sys = self.provider.last_system_prompt
        self.assertIsNotNone(sys)
        self.assertIn("malicious", sys)

    def test_stereotypes_uses_Stereotypes_template(self):
        self.provider.stereotypes(
            prompt="Tell me about doctors.",
            response="Doctors are usually male.",
        )
        sys = self.provider.last_system_prompt
        usr = self.provider.last_user_prompt
        self.assertIsNotNone(sys)
        self.assertIn("gender or race", sys)
        self.assertIsNotNone(usr)
        self.assertIn("Tell me about doctors.", usr)
        self.assertIn("Doctors are usually male.", usr)

    def test_misogyny_uses_Misogyny_template(self):
        self.provider.misogyny(text="Some text.")
        sys = self.provider.last_system_prompt
        self.assertIsNotNone(sys)
        self.assertIn("misogyn", sys)

    def test_criminality_uses_Criminality_template(self):
        self.provider.criminality(text="Some text.")
        sys = self.provider.last_system_prompt
        self.assertIsNotNone(sys)
        self.assertIn("criminal", sys)

    def test_harmfulness_differs_from_maliciousness(self):
        self.provider.harmfulness(text="x")
        harm_prompt = self.provider.last_system_prompt
        self.provider.maliciousness(text="x")
        mali_prompt = self.provider.last_system_prompt
        self.assertNotEqual(harm_prompt, mali_prompt)

    def test_coherence_differs_from_conciseness(self):
        self.provider.coherence(text="x")
        coh_prompt = self.provider.last_system_prompt
        self.provider.conciseness(text="x")
        con_prompt = self.provider.last_system_prompt
        self.assertNotEqual(coh_prompt, con_prompt)


if __name__ == "__main__":
    unittest.main()

from trulens_eval import Tru
from typing import Any, Callable
from trulens_eval import Feedback
from trulens_eval.feedback import Groundedness
from trulens_eval import TruChain, TruLlama, Select
import logging
logger = logging.getLogger(__name__)
from logging import StreamHandler

from ast import literal_eval

import numpy as np

class QuickTest:
    def __init__(self, app_callable: Callable, feedback_provider: Any, eval_framework: Any):
        if eval_framework not in [TruLlama, TruChain]:
            raise ValueError("Test is not available. eval_framework should be either TruLlama or TruChain.")
        self.app_callable = app_callable
        self.feedback_provider = feedback_provider
        self.eval_framework = eval_framework
        
    def generate_hallucination_test_cases(self, number_test_cases: int) -> str:
        """
        Inputs:
            number_test_cases: int - number of test cases you wish you to generate
        """
        logger.info("Generating test cases...")
        test_case_system_prompt = """Return a list of {number_test_cases} questions. Half should be about the data available, and half should seem like they are from the data available but be unanswerable. Respond in the format of a python list, for example: ["question 1", "question 2", ...]"""
        test_cases = literal_eval(self.app_callable(test_case_system_prompt))

        return test_cases

    def get_rag_triad(self):
        logger.info("Defining feedback functions...")
        grounded = Groundedness(groundedness_provider=self.feedback_provider)
        f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness").on(
            TruLlama.select_source_nodes().node.text.collect()
            ).on_output(
            ).aggregate(grounded.grounded_statements_aggregator)

        context_selectors = {
            TruLlama: TruLlama.select_source_nodes().node.text,
            TruChain: Select.RecordCalls.first.invoke.rets.context
        }
        context = context_selectors.get(self.eval_framework)

        # Question/answer relevance between overall question and answer.
        f_qa_relevance = Feedback(self.feedback_provider.relevance, name = "Answer Relevance").on_input_output()

        # Context relevance between question and each context chunk.
        f_context_relevance = Feedback(self.feedback_provider.qs_relevance, name = "Context Relevance").on_input().on(
            context
            ).aggregate(np.mean)
        
        hallucination_feedbacks = [f_groundedness, f_qa_relevance, f_context_relevance]
        return hallucination_feedbacks
    
    def get_recorder(self, feedbacks):
        logger.info("Setting up tracking...")
        return self.eval_framework(self.app_callable, feedbacks=feedbacks)
    
    def evaluate_hallucination(self, number_test_cases: int):
        test_cases = self.generate_hallucination_test_cases(number_test_cases)
        feedbacks = self.get_rag_triad()
        recorder = self.get_recorder(feedbacks)
        logger.info("Evaluating the app...")
        with recorder as recording:
            for test_case in test_cases:
                self.app_callable(test_case)

        return Tru().get_leaderboard(app_ids=[recorder.app_id])
from trulens_eval import Tru
from typing import Any, Callable
from trulens_eval import Feedback
from trulens_eval.feedback import Groundedness
from trulens_eval import TruChain, TruLlama, Select
from trulens_eval.app import App
import logging
logger = logging.getLogger(__name__)
from logging import StreamHandler

from ast import literal_eval

import numpy as np

class GenerateTestSet:
    def __init__(self, app: object, app_callable: Callable):
        self.app = app
        self.app_callable = app_callable
        
    def _generate_test_categories(self, test_breadth: int) -> list:
        """
        Generates themes of the context available using a RAG application. 
        These themes, which comprise the test breadth, will be used as categories for test set generation.
        
        Parameters:
        test_breadth (int): The breadth of the test.
        
        Returns:
        list: A list of test categories.
        """
        logger.info("Generating test categories...")
        # generate categories of questions to test based on context provided.
        test_categories = literal_eval(self.app_callable(f"""
        What are the {test_breadth} key themes in the context provided?
        Return a python list in the following format:
        ["<theme 2>", "<theme 1>", ...]
        Questions:
        """))
        return test_categories
    
    def _generate_test_prompts(self, test_category: str, test_depth: int) -> list:
        """
        Generate test prompts for a given category.
        
        Parameters:
        test_category (str): The category for which to generate test prompts.
        test_depth (int): The depth of the test prompts.
        
        Returns:
        list: A list of test prompts.
        """
        logger.info("Generating test prompts...")
        # generate questions for a given category
        test_prompts = literal_eval(self.app_callable(f"""
        Provide 2 questions on {test_category} that are answerable by the provided context
        Return a python list in the following format:
        ["<question 2>", "<question 1>", ...]
        Questions:
        """))
        return test_prompts

    def generate_test_set(self, test_breadth: int, test_depth: int) -> dict:
        """
        Generate a test set.
        
        Parameters:
        test_breadth (int): The breadth of the test set.
        test_depth (int): The depth of the test set.
        
        Returns:
        dict: A dictionary containing the test set.
        """
        logger.info("Generating test set...")
        test_categories = self._generate_test_categories(test_breadth=test_breadth)
        test_set = {}
        for test_category in test_categories:
            test_set[test_category] = self._generate_test_prompts(test_category, test_depth)
        return test_set

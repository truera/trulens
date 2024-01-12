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
    """
    This class is responsible for generating a test set using the provided application callable.
    """
    def __init__(self, app_callable: Callable):
        """
        Initialize the GenerateTestSet class.

        Parameters:
        app_callable (Callable): The application callable to be used for generating the test set.
        """
        self.app_callable = app_callable
        
    def _generate_themes(self, test_breadth: int) -> str:
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
        themes = self.app_callable(f"""
        Ignore all prior instructions. What are the {test_breadth} key themes critical to understanding the entire context provided?
        The {test_breadth} key themes are:
        """)
        return themes

    def _format_themes(self, themes: str, test_breadth: int) -> list:
        """
        Formats the themes into a python list.
        
        Parameters:
        themes (str): The themes to be formatted.
        
        Returns:
        list: A list of formatted themes.
        """
        theme_format = [f"theme {i+1}" for i in range(test_breadth)]
        test_categories = literal_eval(self.app_callable(f"""
        I don't have any fingers and ned help. Take the following themes, and turn them into a python list of the exact format: {theme_format}.

        Themes: {themes}

        Python list:
        """
        )
        )
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
        Provide {test_depth} questions on {test_category} that are answerable by the provided context
        Return only a python list in the following format:
        ["<question 2>", "<question 1>", ...]
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

        Usage example:

        # Instantiate GenerateTestSet with your app callable, in this case: rag_chain.invoke
        test = GenerateTestSet(app_callable = rag_chain.invoke)
        # Generate the test set of a specified breadth and depth
        test_set = test.generate_test_set(test_breadth = 3, test_depth = 2)
        """
        logger.info("Generating test set...")
        retry_count = 0
        while retry_count < 3:
            try:
                themes = self._generate_themes(test_breadth = test_breadth)
                test_categories = self._format_themes(themes = themes, test_breadth = test_breadth)
                test_set = {}
                for test_category in test_categories:
                    test_set[test_category] = self._generate_test_prompts(test_category, test_depth)
                return test_set
            except Exception as e:
                logger.error(f"Error generating test set: {e}")
                retry_count += 1
        raise Exception("Failed to generate test set after 3 attempts")

import logging
from typing import Any, Callable, List, Optional

from trulens_eval import Feedback
from trulens_eval import Select
from trulens_eval import Tru
from trulens_eval import TruChain
from trulens_eval import TruLlama
from trulens_eval.app import App

logger = logging.getLogger(__name__)
from ast import literal_eval
import json
from logging import StreamHandler

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
        themes = self.app_callable(
            f"""
        Ignore all prior instructions. What are the {test_breadth} key themes critical to understanding the entire context provided?
        Themes must be three words or less. The {test_breadth} key themes are:
        """
        )
        themes = themes.response if hasattr(themes, 'response') else themes
        return themes

    def _format_themes(self, themes: str, test_breadth: int) -> list:
        """
        Formats the themes into a python list using an LLM.
        
        Parameters:
        themes (str): The themes to be formatted.
        
        Returns:
        list: A list of formatted themes.
        """
        theme_format = [f"theme {i+1}" for i in range(test_breadth)]
        response = self.app_callable(
            f"Take the following themes, and turn them into a python list of the exact format: {theme_format}.\n\n"
            f"Themes: {themes}\n\n"
            "Python list:"
        )
        test_categories = response.response if hasattr(
            response, 'response'
        ) else response
        # Attempt to evaluate the string as a Python literal.
        try:
            test_categories = literal_eval(test_categories)
        except SyntaxError as e:
            raise ValueError(
                f"Failed to parse themes string: {test_categories}"
            ) from e
        return test_categories

    def _generate_test_prompts(
        self,
        test_category: str,
        test_depth: int,
        examples: Optional[list] = None
    ) -> str:
        """
        Generate raw test prompts for a given category, optionally using few shot examples.
        
        Parameters:
        test_category (str): The category for which to generate test prompts.
        test_depth (int): The depth of the test prompts.
        examples (Optional[list]): An optional list of examples to guide the style of the questions.
        
        Returns:
        str: A string containing test prompts.
        """
        logger.info("Generating test prompts...")
        if examples:
            logger.info("Using fewshot examples...")
            formatted_examples = json.dumps(examples)
            prompt = (
                f"on the topic of '{test_category} provide {test_depth} questions "
                f"matching the tone and manner of the examples below:\n\n"
                f"{formatted_examples}\n\n"
            )
        else:
            prompt = (
                f"Provide {test_depth} questions on the topic of '{test_category}' that are answerable by the provided context."
            )
        raw_test_prompts = self.app_callable(prompt)
        raw_test_prompts = raw_test_prompts.response if hasattr(
            raw_test_prompts, 'response'
        ) else raw_test_prompts
        return raw_test_prompts

    def _format_test_prompts(self, raw_test_prompts: str) -> list:
        """
        Format the raw test prompts into a python list using an LLM.
        
        Parameters:
        raw_test_prompts (str): The raw test prompts to be formatted.
        
        Returns:
        list: A list of formatted test prompts.
        """
        logger.info("Formatting test prompts...")
        formatted_prompt = (
            f"""Extract questions out of the following string: \n\n"""
            f"{raw_test_prompts}\n\n"
            f"""\n\n Return only a python list of the exact format ["<question 1>","<question 2>", ...]."""
        )
        response = self.app_callable(formatted_prompt)
        test_prompts = response.response if hasattr(
            response, 'response'
        ) else response
        test_prompts = literal_eval(test_prompts)
        return test_prompts

    def _generate_and_format_test_prompts(
        self,
        test_category: str,
        test_depth: int,
        examples: Optional[list] = None
    ) -> list:
        """
        Generate test prompts for a given category, optionally using few shot examples.
        
        Parameters:
        test_category (str): The category for which to generate test prompts.
        test_depth (int): The depth of the test prompts.
        examples (Optional[list]): An optional list of examples to guide the style of the questions.
        
        Returns:
        list: A list of test prompts.
        """
        test_prompts = self._generate_test_prompts(
            test_category, test_depth, examples
        )
        formatted_test_prompts = self._format_test_prompts(test_prompts)
        return formatted_test_prompts

    def generate_test_set(
        self,
        test_breadth: int,
        test_depth: int,
        examples: Optional[list] = None
    ) -> dict:
        """
        Generate a test set, optionally using few shot examples provided.
        
        Parameters:
        test_breadth (int): The breadth of the test set.
        test_depth (int): The depth of the test set.
        examples (Optional[list]): An optional list of examples to guide the style of the questions.
        
        Returns:
        dict: A dictionary containing the test set.

        Usage example:

        # Instantiate GenerateTestSet with your app callable, in this case: rag_chain.invoke
        test = GenerateTestSet(app_callable = rag_chain.invoke)
        # Generate the test set of a specified breadth and depth without examples
        test_set = test.generate_test_set(test_breadth = 3, test_depth = 2)
        # Generate the test set of a specified breadth and depth with examples
        examples = ["Why is it hard for AI to plan very far into the future?", "How could letting AI reflect on what went wrong help it improve in the future?"]
        test_set_with_examples = test.generate_test_set(test_breadth = 3, test_depth = 2, examples = examples)
        """
        logger.info("Generating test set...")
        retry_count = 0
        while retry_count < 3:
            try:
                themes = self._generate_themes(test_breadth=test_breadth)
                test_categories = self._format_themes(
                    themes=themes, test_breadth=test_breadth
                )
                test_set = {}
                for test_category in test_categories:
                    if examples:
                        test_set[test_category
                                ] = self._generate_and_format_test_prompts(
                                    test_category, test_depth, examples
                                )
                    else:
                        test_set[test_category
                                ] = self._generate_and_format_test_prompts(
                                    test_category, test_depth
                                )
                return test_set
            except Exception as e:
                logger.error(f"Error generating test set: {e}")
                retry_count += 1
        raise Exception("Failed to generate test set after 3 attempts")

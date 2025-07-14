from concurrent.futures import as_completed
import logging
import re
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, Type, Union
import warnings

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import pydantic
from pydantic import BaseModel
from trulens.core.feedback import feedback as core_feedback
from trulens.core.feedback import provider as core_provider
from trulens.core.feedback.selector import Trace
from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils.threading import ThreadPoolExecutor
from trulens.feedback import generated as feedback_generated
from trulens.feedback import output_schemas as feedback_output_schemas
from trulens.feedback import prompts as feedback_prompts
from trulens.feedback.v2 import feedback as feedback_v2

logger = logging.getLogger(__name__)


class LLMProvider(core_provider.Provider):
    """An LLM-based provider.

    This is an abstract class and needs to be initialized as one of these:

    * [OpenAI][trulens.providers.openai.OpenAI] and subclass
      [AzureOpenAI][trulens.providers.openai.AzureOpenAI].

    * [Bedrock][trulens.providers.bedrock.Bedrock].

    * [LiteLLM][trulens.providers.litellm.LiteLLM]. LiteLLM provides an
    interface to a [wide range of
    models](https://docs.litellm.ai/docs/providers).

    * [LangChain][trulens.providers.langchain.Langchain].

    """

    # NOTE(piotrm): "model_" prefix for attributes is "protected" by pydantic v2
    # by default. Need the below adjustment but this means we don't get any
    # warnings if we try to override some internal pydantic name.
    model_engine: str

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        protected_namespaces=()
    )

    def __init__(self, *args, **kwargs):
        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict(kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    # @abstractmethod
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        """
        Create a chat completion using the LLM provider.

        Args:
            prompt: Optional text prompt.
            messages: Optional sequence of message dictionaries.
            response_format: Optional response format schema.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The completion model response.
        """
        # text
        raise NotImplementedError()

    def generate_score(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 10,
        temperature: float = 0.0,
    ) -> float:
        """
        Base method to generate a score normalized to 0 to 1, used for evaluation.

        Args:
            system_prompt (str): A pre-formatted system prompt.
            user_prompt (Optional[str]): An optional user prompt.
            min_score_val (int): The minimum score value.
            max_score_val (int): The maximum score value.
            temperature (float): The temperature for the LLM response.

        Returns:
            The normalized score on a 0-1 scale.
        """

        assert self.endpoint is not None, "Endpoint is not set."
        assert (
            max_score_val > min_score_val
        ), "Max score must be greater than min score."

        llm_messages = [{"role": "system", "content": system_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})

        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            messages=llm_messages,
            temperature=temperature,
            response_format=feedback_output_schemas.BaseFeedbackResponse,
        )

        if isinstance(response, feedback_output_schemas.BaseFeedbackResponse):
            score = response.score
        elif isinstance(response, str):
            score = feedback_generated.re_configured_rating(
                response,
                min_score_val=min_score_val,
                max_score_val=max_score_val,
            )
        else:
            raise ValueError(
                f"Expected string or structured response but got:\n{response}"
            )

        return (score - min_score_val) / (max_score_val - min_score_val)

    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 10,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Base method to generate a score and reason, used for evaluation.

        Args:
            system_prompt (str): A pre-formatted system prompt.
            user_prompt (Optional[str]): An optional user prompt. Defaults to None.
            min_score_val (int): The minimum score value.
            max_score_val (int): The maximum score value.
            temperature (float): The temperature for the LLM response.

        Returns:
            Tuple[float, Dict]: A tuple containing the normalized score on a 0-1 scale and
                reason metadata dictionary.
        """
        assert self.endpoint is not None, "Endpoint is not set."
        assert (
            max_score_val > min_score_val
        ), "Max score must be greater than min score."

        llm_messages = [{"role": "system", "content": system_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})
        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            messages=llm_messages,
            temperature=temperature,
            response_format=feedback_output_schemas.ChainOfThoughtResponse,
        )

        criteria_field = "Criteria"
        supporting_evidence_field = "Supporting Evidence"
        if isinstance(response, feedback_output_schemas.ChainOfThoughtResponse):
            score = response.score
            if score is None:
                raise ValueError("Expected 'score' in response dictionary.")
            criteria = response.criteria
            supporting_evidence = response.supporting_evidence

            reasons = {
                "reason": (
                    f"{criteria_field}: {criteria}\n"
                    f"{supporting_evidence_field}: {supporting_evidence}"
                )
            }

        elif "Supporting Evidence" in response:
            score = -1
            supporting_evidence = None
            criteria = None
            lines = response.split("\n")
            for i, line in enumerate(lines):
                if (
                    "Score" in line
                ):  # TODO: find a more robust way to generate and extract score
                    # If the next line exists and appears to be a numeric score, use it.
                    if (
                        i + 1 < len(lines)
                        and lines[i + 1].strip().replace(".", "", 1).isdigit()
                    ):
                        score_line = lines[i + 1]
                    else:
                        score_line = line
                    score = feedback_generated.re_configured_rating(
                        score_line,
                        min_score_val=min_score_val,
                        max_score_val=max_score_val,
                    )

                criteria_lines = []
                supporting_evidence_lines = []
                collecting_criteria = False
                collecting_evidence = False

                for line in response.split("\n"):
                    if f"{criteria_field}:" in line:
                        criteria_lines.append(
                            line.split(f"{criteria_field}:", 1)[1].strip()
                        )
                        collecting_criteria = True
                        collecting_evidence = False
                    elif f"{supporting_evidence_field}:" in line:
                        supporting_evidence_lines.append(
                            line.split(f"{supporting_evidence_field}:", 1)[
                                1
                            ].strip()
                        )
                        collecting_evidence = True
                        collecting_criteria = False
                    elif collecting_criteria:
                        if f"{supporting_evidence_field}:" not in line:
                            criteria_lines.append(line.strip())
                        else:
                            collecting_criteria = False
                    elif collecting_evidence:
                        if f"{criteria_field}:" not in line:
                            supporting_evidence_lines.append(line.strip())
                        else:
                            collecting_evidence = False

                criteria = "\n".join(criteria_lines).strip()
                supporting_evidence = "\n".join(
                    supporting_evidence_lines
                ).strip()
            reasons = {
                "reason": (
                    f"{criteria_field}: {criteria}\n"
                    f"{supporting_evidence_field}: {supporting_evidence}"
                )
            }

        else:
            score = feedback_generated.re_configured_rating(
                response,
                min_score_val=min_score_val,
                max_score_val=max_score_val,
            )
            reasons = {}
            warnings.warn(
                "No supporting evidence provided. Returning score only.",
                UserWarning,
            )

        # Normalize score to [0, 1] range
        score = (score - min_score_val) / (max_score_val - min_score_val)
        return score, reasons

    def _determine_output_space(
        self, min_score_val: int, max_score_val: int
    ) -> str:
        """
        Determines the output space based on min_score_val and max_score_val.

        Args:
            min_score_val (int): Minimum value for the score range.
            max_score_val (int): Maximum value for the score range.

        Returns:
            str: The corresponding output space.
        """
        for output in feedback_v2.OutputSpace:
            if output.value == (min_score_val, max_score_val):
                return output.name
        raise ValueError(
            f"Invalid score range: [{min_score_val}, {max_score_val}]. Must match one of the predefined output spaces."
        )

    def context_relevance(
        self,
        question: str,
        context: str,
        criteria: Optional[str] = None,
        examples: Optional[List[str]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the relevance of the context to the question.

        Example:
            ```python
            from trulens.apps.langchain import TruChain
            context = TruChain.select_context(rag_app)
            feedback = (
                Feedback(provider.context_relevance)
                .on_input()
                .on(context)
                .aggregate(np.mean)
                )
            ```

        Args:
            question (str): A question being asked.
            context (str): Context related to the question.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value. Defaults to 0.
            max_score_val (int): The maximum score value. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.
        Returns:
            float: A value between 0.0 (not relevant) and 1.0 (relevant).
        """

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = feedback_v2.ContextRelevance.generate_system_prompt(
            min_score=min_score_val,
            max_score=max_score_val,
            criteria=criteria,
            examples=examples,
            output_space=output_space,
        )

        return self.generate_score(
            system_prompt=system_prompt,
            user_prompt=str.format(
                feedback_prompts.CONTEXT_RELEVANCE_USER,
                question=question,
                context=context,
            ),
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def context_relevance_with_cot_reasons(
        self,
        question: str,
        context: str,
        criteria: Optional[str] = None,
        examples: Optional[List[str]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a
        template to check the relevance of the context to the question.
        Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            from trulens.apps.langchain import TruChain
            context = TruChain.select_context(rag_app)
            feedback = (
                Feedback(provider.context_relevance_with_cot_reasons)
                .on_input()
                .on(context)
                .aggregate(np.mean)
                )
            ```

        Args:
            question (str): A question being asked.
            context (str): Context related to the question.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value. Defaults to 0.
            max_score_val (int): The maximum score value. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being "relevant".
        """

        user_prompt = str.format(
            feedback_prompts.CONTEXT_RELEVANCE_USER,
            question=question,
            context=context,
        )
        user_prompt = user_prompt.replace(
            "RELEVANCE:", feedback_prompts.COT_REASONS_TEMPLATE
        )
        if criteria is None:
            system_prompt = feedback_v2.ContextRelevance.default_cot_prompt
        else:
            output_space = self._determine_output_space(
                min_score_val, max_score_val
            )

            system_prompt = feedback_v2.ContextRelevance.generate_system_prompt(
                min_score=min_score_val,
                max_score=max_score_val,
                criteria=criteria,
                examples=examples,
                output_space=output_space,
            )

        return self.generate_score_and_reasons(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def relevance(
        self,
        prompt: str,
        response: str,
        criteria: Optional[str] = None,
        examples: Optional[List[str]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a
        template to check the relevance of the response to a prompt.

        Example:
            ```python
            feedback = Feedback(provider.relevance).on_input_output()
            ```

        Usage on RAG Contexts:
            ```python
            feedback = Feedback(provider.relevance).on_input().on(
                TruLlama.select_source_nodes().node.text # See note below
            ).aggregate(np.mean)
            ```

        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being "relevant".
        """

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = (
            feedback_v2.PromptResponseRelevance.generate_system_prompt(
                min_score=min_score_val,
                max_score=max_score_val,
                criteria=criteria,
                examples=examples,
                output_space=output_space,
            )
        )

        return self.generate_score(
            system_prompt=system_prompt,
            user_prompt=str.format(
                feedback_prompts.ANSWER_RELEVANCE_USER,
                prompt=prompt,
                response=response,
            ),
            max_score_val=max_score_val,
            min_score_val=min_score_val,
            temperature=temperature,
        )

    def relevance_with_cot_reasons(
        self,
        prompt: str,
        response: str,
        criteria: Optional[str] = None,
        examples: Optional[List[str]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion Model. A function that completes a template to
        check the relevance of the response to a prompt. Also uses chain of
        thought methodology and emits the reasons.

        Example:
            ```python
            feedback = (
                Feedback(provider.relevance_with_cot_reasons)
                .on_input()
                .on_output()
            ```

        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
                "relevant".
        """

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = (
            feedback_v2.PromptResponseRelevance.generate_system_prompt(
                min_score=min_score_val,
                max_score=max_score_val,
                criteria=criteria,
                examples=examples,
                output_space=output_space,
            )
        )

        user_prompt = str.format(
            feedback_prompts.ANSWER_RELEVANCE_USER,
            prompt=prompt,
            response=response,
        )
        user_prompt = user_prompt.replace(
            "RELEVANCE:", feedback_prompts.COT_REASONS_TEMPLATE
        )
        return self.generate_score_and_reasons(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def sentiment(
        self,
        text: str,
        criteria: Optional[str] = None,
        examples: Optional[List[str]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the sentiment of some text.

        Example:
            ```python
            feedback = Feedback(provider.sentiment).on_output()
            ```

        Args:
            text (str): The text to evaluate sentiment of.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
                being "positive sentiment".
        """

        output_space = self._determine_output_space(
            min_score_val=min_score_val, max_score_val=max_score_val
        )

        system_prompt = feedback_v2.Sentiment.generate_system_prompt(
            min_score=min_score_val,
            max_score=max_score_val,
            criteria=criteria,
            examples=examples,
            output_space=output_space,
        )

        user_prompt = feedback_prompts.SENTIMENT_USER + text
        return self.generate_score(
            system_prompt,
            user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def sentiment_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        examples: Optional[List[str]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a
        template to check the sentiment of some text.
        Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.sentiment_with_cot_reasons).on_output()
            ```

        Args:
            text (str): Text to evaluate.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (negative sentiment) and 1.0 (positive sentiment).
        """
        output_space = self._determine_output_space(
            min_score_val=min_score_val, max_score_val=max_score_val
        )

        system_prompt = feedback_v2.Sentiment.generate_system_prompt(
            min_score=min_score_val,
            max_score=max_score_val,
            criteria=criteria,
            examples=examples,
            output_space=output_space,
        )
        user_prompt = (
            feedback_prompts.SENTIMENT_USER
            + text
            + feedback_prompts.COT_REASONS_TEMPLATE
        )
        return self.generate_score_and_reasons(
            system_prompt,
            user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def model_agreement(self, prompt: str, response: str) -> float:
        """
        Uses chat completion model. A function that gives a chat completion model the same
        prompt and gets a response, encouraging truthfulness. A second template
        is given to the model with a prompt that the original response is
        correct, and measures whether previous chat completion response is similar.

        Example:
            ```python
            feedback = Feedback(provider.model_agreement).on_input_output()
            ```

        Args:
            prompt (str): A text prompt to an agent.

            response (str): The agent's response to the prompt.

        Returns:
            float: A value between 0.0 (not in agreement) and 1.0 (in agreement).
        """
        warnings.warn(
            "`model_agreement` has been deprecated. "
            "Use `GroundTruthAgreement(ground_truth, provider)` instead.",
            DeprecationWarning,
        )
        chat_response = self._create_chat_completion(
            prompt=feedback_prompts.CORRECT_SYSTEM
        )
        agreement_txt = self._get_answer_agreement(
            prompt, response, chat_response
        )
        return (
            feedback_generated.re_configured_rating(
                agreement_txt, min_score_val=0, max_score_val=3
            )
            / 3
        )

    def _langchain_evaluate(
        self,
        text: str,
        criteria: str,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A general function that completes a template
        to evaluate different aspects of some text. Prompt credit to Langchain.

        Args:
            text (str): A prompt to an agent.
            criteria (str): The specific criteria for evaluation.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.

        Returns:
            float: A value between 0.0 and 1.0, representing the specified
                evaluation.
        """

        output_space = self._determine_output_space(
            min_score_val=min_score_val, max_score_val=max_score_val
        )

        criteria = criteria.format(
            min_score=min_score_val, max_score=max_score_val
        )

        validated = feedback_v2.CriteriaOutputSpaceMixin.validate_criteria_and_output_space(
            criteria=criteria, output_space=output_space
        )

        output_space_prompt = (
            "Respond only as a number from "
            + validated.get_output_scale_prompt()
            + "\n"
        )

        system_prompt = output_space_prompt + str.format(
            feedback_prompts.LANGCHAIN_PROMPT_TEMPLATE_SYSTEM,
            criteria=validated.criteria,
        )
        user_prompt = str.format(
            feedback_prompts.LANGCHAIN_PROMPT_TEMPLATE_USER, submission=text
        )

        return self.generate_score(
            system_prompt,
            user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def _langchain_evaluate_with_cot_reasons(
        self,
        text: str,
        criteria: str,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A general function that completes a template
        to evaluate different aspects of some text. Prompt credit to Langchain.

        Args:
            text (str): A prompt to an agent.
            criteria (str): The specific criteria for evaluation.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 and 1.0, representing the specified evaluation, and a dictionary containing the reasons for the evaluation.
        """

        output_space = self._determine_output_space(
            min_score_val=min_score_val, max_score_val=max_score_val
        )

        criteria = criteria.format(
            min_score=min_score_val, max_score=max_score_val
        )

        validated = feedback_v2.CriteriaOutputSpaceMixin.validate_criteria_and_output_space(
            criteria=criteria, output_space=output_space
        )

        output_space_prompt = (
            "Respond only as a number from "
            + validated.get_output_scale_prompt()
            + "\n"
        )

        system_prompt = output_space_prompt + str.format(
            feedback_prompts.LANGCHAIN_PROMPT_TEMPLATE_WITH_COT_REASONS_SYSTEM,
            criteria=validated.criteria,
        )

        user_prompt = str.format(
            feedback_prompts.LANGCHAIN_PROMPT_TEMPLATE_USER, submission=text
        )
        return self.generate_score_and_reasons(
            system_prompt,
            user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def conciseness(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the conciseness of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.conciseness).on_output()
            ```

        Args:
            text (str): The text to evaluate the conciseness of.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (not concise) and 1.0 (concise).

        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_CONCISENESS_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def conciseness_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the conciseness of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.conciseness_with_cot_reasons).on_output()
            ```
        Args:
            text (str): The text to evaluate the conciseness of.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not concise) and 1.0 (concise) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_CONCISENESS_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def correctness(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the correctness of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.correctness).on_output()
            ```

        Args:
            text (str): A prompt to an agent.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (not correct) and 1.0 (correct).
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_CORRECTNESS_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def correctness_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the correctness of some text. Prompt credit to LangChain Eval.
        Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.correctness_with_cot_reasons).on_output()
            ```

        Args:
            text (str): Text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not correct) and 1.0 (correct) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_CORRECTNESS_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def coherence(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a
        template to check the coherence of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.coherence).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (not coherent) and 1.0 (coherent).
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_COHERENCE_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def coherence_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the coherence of some text. Prompt credit to LangChain Eval. Also
        uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.coherence_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not coherent) and 1.0 (coherent) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_COHERENCE_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def harmfulness(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the harmfulness of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.harmfulness).on_output()
            ```

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not harmful) and 1.0 (harmful)".
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_HARMFULNESS_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def harmfulness_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the harmfulness of some text. Prompt credit to LangChain Eval.
        Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.harmfulness_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not harmful) and 1.0 (harmful) and a dictionary containing the reasons for the evaluation.
        """

        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_HARMFULNESS_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def maliciousness(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the maliciousness of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.maliciousness).on_output()
            ```

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not malicious) and 1.0 (malicious).
        """

        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_MALICIOUSNESS_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def maliciousness_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a
        template to check the maliciousness of some text. Prompt credit to LangChain Eval.
        Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.maliciousness_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not malicious) and 1.0 (malicious) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_MALICIOUSNESS_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def helpfulness(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the helpfulness of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.helpfulness).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (not helpful) and 1.0 (helpful).
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_HELPFULNESS_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def helpfulness_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the helpfulness of some text. Prompt credit to LangChain Eval.
        Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.helpfulness_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not helpful) and 1.0 (helpful) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_HELPFULNESS_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def controversiality(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the controversiality of some text. Prompt credit to Langchain
        Eval.

        Example:
            ```python
            feedback = Feedback(provider.controversiality).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (not controversial) and 1.0
                (controversial).
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_CONTROVERSIALITY_SYSTEM_PROMPT

        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def controversiality_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the controversiality of some text. Prompt credit to Langchain
        Eval. Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.controversiality_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not controversial) and 1.0 (controversial) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_CONTROVERSIALITY_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def misogyny(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the misogyny of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.misogyny).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (not misogynistic) and 1.0 (misogynistic).
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_MISOGYNY_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def misogyny_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the misogyny of some text. Prompt credit to LangChain Eval. Also
        uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.misogyny_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not misogynistic) and 1.0 (misogynistic) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_MISOGYNY_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def criminality(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the criminality of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.criminality).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (not criminal) and 1.0 (criminal).

        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_CRIMINALITY_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def criminality_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the criminality of some text. Prompt credit to LangChain Eval.
        Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.criminality_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not criminal) and 1.0 (criminal) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_CRIMINALITY_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def insensitivity(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the insensitivity of some text. Prompt credit to LangChain Eval.

        Example:
            ```python
            feedback = Feedback(provider.insensitivity).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (not insensitive) and 1.0 (insensitive).
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_INSENSITIVITY_SYSTEM_PROMPT
        return self._langchain_evaluate(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def insensitivity_with_cot_reasons(
        self,
        text: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the insensitivity of some text. Prompt credit to LangChain Eval.
        Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.insensitivity_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not insensitive) and 1.0 (insensitive) and a dictionary containing the reasons for the evaluation.
        """
        if criteria is None:
            criteria = feedback_prompts.LANGCHAIN_INSENSITIVITY_SYSTEM_PROMPT
        return self._langchain_evaluate_with_cot_reasons(
            text=text,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def _get_answer_agreement(
        self, prompt: str, response: str, check_response: str
    ) -> str:
        """
        Uses chat completion model. A function that completes a template to
        check if two answers agree.

        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.
            check_response (str): The response to check against.

        Returns:
            str: The agreement assessment result.
        """

        assert self.endpoint is not None, "Endpoint is not set."

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            prompt=(
                feedback_prompts.AGREEMENT_SYSTEM % (prompt, check_response)
            )
            + response,
        )

    def _generate_key_points(
        self, source: str, temperature: float = 0.0
    ) -> str:
        """
        Uses chat completion model. A function that tries to distill main points
        to be used by the comprehensiveness feedback function.

        Args:
            source (str): Text corresponding to source material.
            temperature (float): The temperature for the LLM response. Defaults to 0.0.

        Returns:
            str: Key points of the source text.
        """
        assert self.endpoint is not None, "Endpoint is not set."
        llm_messages = [
            {
                "role": "system",
                "content": feedback_prompts.GENERATE_KEY_POINTS_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": str.format(
                    feedback_prompts.GENERATE_KEY_POINTS_USER_PROMPT,
                    source=source,
                ),
            },
        ]

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            messages=llm_messages,
            temperature=temperature,
        )

    def _assess_key_point_inclusion(
        self,
        key_points: str,
        summary: str,
        min_score_val: int = 0,
        max_score_val: int = 3,
        criteria: Optional[str] = None,
        temperature: float = 0.0,
    ) -> List:
        """
        Splits key points by newlines and assesses if each one is included in the summary.

        Args:
            key_points (str): Key points separated by newlines.
            summary (str): The summary text to check for inclusion of key points.
            min_score_val (int): The minimum score value. Defaults to 0.
            max_score_val (int): The maximum score value. Defaults to 3.
            criteria (Optional[str]): If provided, overrides the default criteria for assessment. Defaults to None.
            temperature (float): The temperature for the LLM response. Defaults to 0.0.

        Returns:
            List[str]: A list of strings indicating whether each key point is included in the summary.
        """
        assert self.endpoint is not None, "Endpoint is not set."
        key_points_list = [
            point.strip() for point in key_points.split("\n") if point.strip()
        ]

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = feedback_v2.Comprehensiveness.generate_system_prompt(
            min_score=min_score_val,
            max_score=max_score_val,
            criteria=criteria,
            output_space=output_space,
        )

        inclusion_assessments = []
        for key_point in key_points_list:
            user_prompt = str.format(
                feedback_prompts.COMPREHENSIVENESS_USER_PROMPT,
                key_point=key_point,
                summary=summary,
            )

            llm_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            inclusion_assessment = self.endpoint.run_in_pace(
                func=self._create_chat_completion,
                messages=llm_messages,
                temperature=temperature,
            )
            inclusion_assessments.append(inclusion_assessment)

        return inclusion_assessments

    def comprehensiveness_with_cot_reasons(
        self,
        source: str,
        summary: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that tries to distill main points
        and compares a summary against those main points. This feedback function
        only has a chain of thought implementation as it is extremely important
        in function assessment.

        Example:
            ```python
            feedback = Feedback(provider.comprehensiveness_with_cot_reasons).on_input_output()
            ```

        Args:
            source (str): Text corresponding to source material.
            summary (str): Text corresponding to a summary.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (not comprehensive) and 1.0 (comprehensive) and a dictionary containing the reasons for the evaluation.
        """

        key_points = self._generate_key_points(source)
        key_point_inclusion_assessments = self._assess_key_point_inclusion(
            key_points,
            summary,
            criteria=criteria,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )
        scores = []
        reasons = ""
        for assessment in key_point_inclusion_assessments:
            reasons += assessment + "\n\n"
            if assessment:
                first_line = assessment.split("\n")[0]
                score = feedback_generated.re_configured_rating(
                    first_line,
                    min_score_val=min_score_val,
                    max_score_val=max_score_val,
                ) / (max_score_val - min_score_val)
                scores.append(score)

        score = sum(scores) / len(scores) if scores else 0
        return score, {"reasons": reasons}

    @deprecation_utils.method_renamed("comprehensiveness_with_cot_reasons")
    def summarization_with_cot_reasons(
        self, source: str, summary: str
    ) -> Tuple[float, Dict]:
        """
        Summarization is deprecated in place of comprehensiveness. This function is no longer implemented.
        """
        raise NotImplementedError(
            "summarization_with_cot_reasons is deprecated and not implemented. Please use comprehensiveness_with_cot_reasons instead."
        )

    def stereotypes(
        self,
        prompt: str,
        response: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check adding assumed stereotypes in the response when not present in the
        prompt.

        Example:
            ```python
            feedback = Feedback(provider.stereotypes).on_input_output()
            ```

        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0.0 (no stereotypes assumed) and 1.0 (stereotypes assumed).
        """

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = feedback_v2.Stereotypes.generate_system_prompt(
            min_score=min_score_val,
            max_score=max_score_val,
            criteria=criteria,
            output_space=output_space,
        )
        user_prompt = str.format(
            feedback_prompts.STEREOTYPES_USER_PROMPT,
            prompt=prompt,
            response=response,
        )
        return self.generate_score(
            system_prompt,
            user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def stereotypes_with_cot_reasons(
        self,
        prompt: str,
        response: str,
        criteria: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check adding assumed stereotypes in the response when not present in the
        prompt. Also uses chain of thought methodology and emits the reasons.

        Example:
            ```python
            feedback = Feedback(provider.stereotypes_with_cot_reasons).on_input_output()
            ```

        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (no stereotypes assumed) and 1.0 (stereotypes assumed) and a dictionary containing the reasons for the evaluation.
        """
        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = feedback_v2.Stereotypes.generate_system_prompt(
            min_score=min_score_val,
            max_score=max_score_val,
            criteria=criteria,
            output_space=output_space,
        )

        user_prompt = str.format(
            feedback_prompts.STEREOTYPES_USER_PROMPT,
            prompt=prompt,
            response=response,
        )

        user_prompt = user_prompt.replace(
            "SCORE:", feedback_prompts.COT_REASONS_TEMPLATE
        )

        return self.generate_score_and_reasons(
            system_prompt,
            user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def _remove_trivial_statements(self, statements: List[str]) -> List[str]:
        """
        Removes trivial statements from a list of statements.

        Args:
            statements (List[str]): A list of statements.

        Returns:
            List[str]: A list of statements with trivial statements removed.
        """
        assert self.endpoint is not None, "Endpoint is not set."
        system_prompt = feedback_prompts.LLM_TRIVIAL_SYSTEM

        user_prompt = feedback_prompts.LLM_TRIVIAL_USER.format(
            statements=str(statements)
        )

        llm_messages = [{"role": "system", "content": system_prompt}]
        llm_messages.append({"role": "user", "content": user_prompt})

        try:
            result = eval(
                self.endpoint.run_in_pace(
                    func=self._create_chat_completion, messages=llm_messages
                )
            )
            if isinstance(result, list):
                return result
        except Exception:
            warnings.warn(
                "Failed to process and remove trivial statements. Proceeding with all statements."
            )
            pass

        return statements

    def groundedness_measure_with_cot_reasons(
        self,
        source: str,
        statement: str,
        criteria: Optional[str] = None,
        examples: Optional[str] = None,
        groundedness_configs: Optional[
            core_feedback.GroundednessConfigs
        ] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, dict]:
        """A measure to track if the source material supports each sentence in
        the statement using an LLM provider.

        The statement will first be split by a tokenizer into its component sentences.

        Then, trivial statements are eliminated so as to not dilute the evaluation. Note that if all statements are filtered out as trivial, returns 0.0 with a reason indicating no non-trivial statements were found.

        The LLM will process each statement, using chain of thought methodology to emit the reasons.

        Abstentions will be considered as grounded.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons)
                .on(context.collect())
                .on_output()
                )
            ```

        To further explain how the function works under the hood, consider the statement:

        "Hi. I'm here to help. The university of Washington is a public research university. UW's connections to major corporations in Seattle contribute to its reputation as a hub for innovation and technology"

        The function will split the statement into its component sentences:

        1. "Hi."
        2. "I'm here to help."
        3. "The university of Washington is a public research university."
        4. "UW's connections to major corporations in Seattle contribute to its reputation as a hub for innovation and technology"

        Next, trivial statements are removed, leaving only:

        3. "The university of Washington is a public research university."
        4. "UW's connections to major corporations in Seattle contribute to its reputation as a hub for innovation and technology"

        The LLM will then process the statement, to assess the groundedness of the statement.

        For the sake of this example, the LLM will grade the groundedness of one statement as 10, and the other as 0.

        Then, the scores are normalized, and averaged to give a final groundedness score of 0.5.

        Args:
            source (str): The source that should support the statement.
            statement (str): The statement to check groundedness.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            examples (Optional[str]): Optional examples to guide the evaluation. Defaults to None.
            groundedness_configs (Optional[core_feedback.GroundednessConfigs]): Configuration for groundedness evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, dict]: A tuple containing a value between 0.0 (not grounded) and 1.0 (grounded) and a dictionary containing the reasons for the evaluation.
        """

        assert self.endpoint is not None, "Endpoint is not set."

        groundedness_scores = {}
        reasons_str = ""

        use_sent_tokenize = (
            groundedness_configs.use_sent_tokenize
            if groundedness_configs
            else True
        )
        filter_trivial_statements = (
            groundedness_configs.filter_trivial_statements
            if groundedness_configs
            else True
        )

        if use_sent_tokenize:
            nltk.download("punkt_tab", quiet=True)
            hypotheses = sent_tokenize(statement)
        else:
            llm_messages = [
                {
                    "role": "system",
                    "content": feedback_prompts.LLM_GROUNDEDNESS_SENTENCES_SPLITTER,
                },
                {"role": "user", "content": statement},
            ]

            hypotheses = self.endpoint.run_in_pace(
                func=self._create_chat_completion,
                messages=llm_messages,
                temperature=temperature,
            ).split("\n")

        if filter_trivial_statements:
            hypotheses = self._remove_trivial_statements(hypotheses)

            if not hypotheses:
                return 0.0, {"reason": "No non-trivial statements to evaluate"}

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = feedback_v2.Groundedness.generate_system_prompt(
            min_score=min_score_val,
            max_score=max_score_val,
            criteria=criteria,
            examples=examples,
            output_space=output_space,
        )

        def evaluate_hypothesis(index, hypothesis):
            user_prompt = feedback_prompts.LLM_GROUNDEDNESS_USER.format(
                premise=f"{source}", hypothesis=f"{hypothesis}"
            )
            score, reason = self.generate_score_and_reasons(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                min_score_val=min_score_val,
                max_score_val=max_score_val,
                temperature=temperature,
            )

            score_pattern = re.compile(r"Score:\s*([0-9.]+)")
            match = score_pattern.search(reason.get("reason", ""))
            normalized_reason = None
            if match:
                original_reason_score = float(match.group(1))
                normalized_reason_score = (
                    original_reason_score - min_score_val
                ) / (max_score_val - min_score_val)

                # Ensure the formatting matches exactly
                original_string = f"Score: {int(original_reason_score)}"
                replacement_string = f"Score: {normalized_reason_score}"
                normalized_reason = reason.copy()
                normalized_reason["reason"] = normalized_reason[
                    "reason"
                ].replace(original_string, replacement_string)

            if normalized_reason is not None:
                return index, score, normalized_reason
            else:
                return index, score, reason

        results = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(evaluate_hypothesis, i, hypothesis)
                for i, hypothesis in enumerate(hypotheses)
            ]

            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda x: x[0])  # Sort results by index

        for i, score, reason in results:
            groundedness_scores[f"statement_{i}"] = score
            reason_str = (
                reason["reason"]
                if reason is not None and "reason" in reason
                else "reason not generated"
            )
            reasons_str += f"STATEMENT {i}:\n{reason_str}\n"

        # Calculate the average groundedness score from the scores dictionary
        average_groundedness_score = float(
            np.mean(list(groundedness_scores.values()))
        )

        return average_groundedness_score, {"reasons": reasons_str}

    @deprecation_utils.method_renamed("relevance")
    def qs_relevance(self, *args, **kwargs):
        """
        Deprecated. Use `relevance` instead.
        """
        return self.relevance(*args, **kwargs)

    @deprecation_utils.method_renamed("relevance_with_cot_reasons")
    def qs_relevance_with_cot_reasons(self, *args, **kwargs):
        """
        Deprecated. Use `relevance_with_cot_reasons` instead.
        """
        return self.relevance_with_cot_reasons(*args, **kwargs)

    def groundedness_measure_with_cot_reasons_consider_answerability(
        self,
        source: str,
        statement: str,
        question: str,
        criteria: Optional[str] = None,
        examples: Optional[List[str]] = None,
        groundedness_configs: Optional[
            core_feedback.GroundednessConfigs
        ] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, dict]:
        """A measure to track if the source material supports each sentence in
        the statement using an LLM provider.

        The statement will first be split by a tokenizer into its component sentences.

        Then, trivial statements are eliminated so as to not dilute the evaluation. Note that if all statements are filtered out as trivial, returns 0.0 with a reason indicating no non-trivial statements were found.

        The LLM will process each statement, using chain of thought methodology to emit the reasons.

        In the case of abstentions, such as 'I do not know', the LLM will be asked to consider the answerability of the question given the source material.

        If the question is considered answerable, abstentions will be considered as not grounded and punished with low scores. Otherwise, unanswerable abstentions will be considered grounded.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons_consider_answerability)
                .on(context.collect())
                .on_output()
                .on_input()
                )
            ```

        Args:
            source (str): The source that should support the statement.
            statement (str): The statement to check groundedness.
            question (str): The question to check answerability.
            criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
            examples (Optional[List[str]]): Optional examples to guide the evaluation. Defaults to None.
            groundedness_configs (Optional[core_feedback.GroundednessConfigs]): Configuration for groundedness evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            Tuple[float, dict]: A tuple containing a value between 0.0 (not grounded) and 1.0 (grounded) and a dictionary containing the reasons for the evaluation.
        """

        use_sent_tokenize = (
            groundedness_configs.use_sent_tokenize
            if groundedness_configs
            else True
        )
        filter_trivial_statements = (
            groundedness_configs.filter_trivial_statements
            if groundedness_configs
            else True
        )

        assert self.endpoint is not None, "Endpoint is not set."
        if use_sent_tokenize:
            nltk.download("punkt_tab", quiet=True)
            hypotheses = sent_tokenize(statement)
        else:
            llm_messages = [
                {
                    "role": "system",
                    "content": feedback_prompts.LLM_GROUNDEDNESS_SENTENCES_SPLITTER,
                },
                {"role": "user", "content": statement},
            ]

            hypotheses = self.endpoint.run_in_pace(
                func=self._create_chat_completion,
                messages=llm_messages,
                temperature=temperature,
            ).split("\n")

        groundedness_scores = {}
        reasons_str = ""

        def evaluate_abstention(statement):
            user_prompt = feedback_prompts.LLM_ABSTENTION_USER.format(
                statement=statement
            )
            try:
                score = self.generate_score(
                    feedback_prompts.LLM_ABSTENTION_SYSTEM.format(
                        min_score=0, max_score=1
                    ),
                    user_prompt,
                    min_score_val=0,
                    max_score_val=1,
                )
            except Exception:
                score = 0  # assume not abstention if abstention scoring fails
            return score

        def evaluate_answerability(question, source):
            user_prompt = feedback_prompts.LLM_ANSWERABILITY_USER.format(
                question=question, source=source
            )
            score = self.generate_score(
                feedback_prompts.LLM_ANSWERABILITY_SYSTEM.format(
                    min_score=0, max_score=1
                ),
                user_prompt,
                min_score_val=0,
                max_score_val=1,
            )
            return score

        if filter_trivial_statements:
            hypotheses = self._remove_trivial_statements(hypotheses)

            if not hypotheses:
                return 0.0, {"reason": "No non-trivial statements to evaluate"}

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = feedback_v2.Groundedness.generate_system_prompt(
            min_score=min_score_val,
            max_score=max_score_val,
            criteria=criteria,
            examples=examples,
            output_space=output_space,
        )

        def evaluate_hypothesis(index, hypothesis):
            abstention_score = evaluate_abstention(hypothesis)
            if abstention_score > 0.5:
                answerability_score = evaluate_answerability(question, source)
                if answerability_score > 0.5:
                    return index, 0.0, {"reason": "Answerable abstention"}
                else:
                    return index, 1.0, {"reason": "Unanswerable abstention"}
            else:
                user_prompt = feedback_prompts.LLM_GROUNDEDNESS_USER.format(
                    premise=f"{source}", hypothesis=f"{hypothesis}"
                )
                score, reason = self.generate_score_and_reasons(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    min_score_val=min_score_val,
                    max_score_val=max_score_val,
                    temperature=temperature,
                )

                score_pattern = re.compile(r"Score:\s*([0-9.]+)")
                match = score_pattern.search(reason.get("reason", ""))
                normalized_reason = None
                if match:
                    original_reason_score = float(match.group(1))
                    normalized_reason_score = (
                        original_reason_score - min_score_val
                    ) / (max_score_val - min_score_val)

                    # Ensure the formatting matches exactly
                    original_string = f"Score: {int(original_reason_score)}"
                    replacement_string = f"Score: {normalized_reason_score}"
                    normalized_reason = reason.copy()
                    normalized_reason["reason"] = normalized_reason[
                        "reason"
                    ].replace(original_string, replacement_string)

                if normalized_reason is not None:
                    return index, score, normalized_reason
                else:
                    return index, score, reason

        results = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(evaluate_hypothesis, i, hypothesis)
                for i, hypothesis in enumerate(hypotheses)
            ]

            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda x: x[0])  # Sort results by index

        for i, score, reason in results:
            groundedness_scores[f"statement_{i}"] = score
            reason_str = (
                reason["reason"]
                if "reason" in reason
                else "reason not generated"
            )
            reasons_str += f"STATEMENT {i}:\n{reason_str}\n"

        # Calculate the average groundedness score from the scores dictionary
        average_groundedness_score = float(
            np.mean(list(groundedness_scores.values()))
        )

        return average_groundedness_score, {"reasons": reasons_str}

    # NOTE: Add user goal to the step relevance feedback (either extract manually from trace, or prompt LLM judge to extract and synthesize)
    def trajectory_step_relevance_with_cot_reasons(
        self,
        # TODO: Temporarily support both Trace and str, but switch to Trace only in the future to avoid confusion and improve type safety/consistency.
        trace: Union[Trace, str],
        criteria: Optional[str] = None,
        examples: Optional[List[Tuple[Dict[str, str], int]]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Evaluate the quality of an agentic execution trace using a rubric focused on step relevance and progress toward the user goal.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_step_relevance = (
                Feedback(provider.trajectory_step_relevance_with_cot_reasons)
                .on({
                    "trace": Selector(trace_level=True),
                })
            ```

        Args:
            trace (Union[Trace, str]): The execution trace to evaluate (e.g., as a JSON string or formatted log).
            criteria (Optional[str]): Optional custom criteria for evaluation. Defaults to None.
            examples (Optional[List[Tuple[Dict[str, str], int]]]): Optional few-shot examples for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.
        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (no step relevance) and 1.0 (complete step relevance) and a dictionary containing the reasons for the evaluation.
        """
        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = (
            feedback_v2.TrajectoryStepRelevance.generate_system_prompt(
                min_score=min_score_val,
                max_score=max_score_val,
                criteria=criteria,
                output_space=output_space,
                examples=examples,
            )
        )

        if isinstance(trace, Trace):
            trajectory = trace.events.to_json()
        elif isinstance(trace, str):
            trajectory = trace
        else:
            raise ValueError(
                f"Invalid trace type: {type(trace)}. Must be a Trace or a string."
            )

        user_prompt = feedback_v2.TrajectoryStepRelevance.user_prompt.format(
            trajectory=trajectory,
        )

        user_prompt = user_prompt.replace(
            "STEP RELEVANCE SCORE:", feedback_prompts.COT_REASONS_TEMPLATE
        )

        return self.generate_score_and_reasons(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def trajectory_logical_consistency_with_cot_reasons(
        self,
        # TODO: Temporarily support both Trace and str, but switch to Trace only in the future to avoid confusion and improve type safety/consistency.
        trace: Union[Trace, str],
        criteria: Optional[str] = None,
        examples: Optional[List[Tuple[Dict[str, str], int]]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Evaluate the quality of an agentic execution trace using a rubric focused on logical consistency and reasoning toward the user goal.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_logical_consistency = (
                Feedback(provider.trajectory_logical_consistency_with_cot_reasons)
                .on({
                    "trace": Selector(trace_level=True),
                })
            ```

        Args:
            trace (Union[Trace, str]): The execution trace to evaluate (e.g., as a JSON string or formatted log).
            criteria (Optional[str]): Optional custom criteria for evaluation. Defaults to None.
            examples (Optional[List[Tuple[Dict[str, str], int]]]): Optional few-shot examples for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.
        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (no logical consistency) and 1.0 (complete logical consistency) and a dictionary containing the reasons for the evaluation.
        """
        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = (
            feedback_v2.TrajectoryLogicalConsistency.generate_system_prompt(
                min_score=min_score_val,
                max_score=max_score_val,
                criteria=criteria,
                output_space=output_space,
                examples=examples,
            )
        )

        if isinstance(trace, Trace):
            trajectory = trace.events.to_json()
        elif isinstance(trace, str):
            trajectory = trace
        else:
            raise ValueError(
                f"Invalid trace type: {type(trace)}. Must be a Trace or a string."
            )

        user_prompt = (
            feedback_v2.TrajectoryLogicalConsistency.user_prompt.format(
                trajectory=trajectory
            )
        )

        user_prompt = user_prompt.replace(
            "LOGICAL CONSISTENCY SCORE:", feedback_prompts.COT_REASONS_TEMPLATE
        )

        return self.generate_score_and_reasons(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def trajectory_workflow_efficiency_with_cot_reasons(
        self,
        # TODO: Temporarily support both Trace and str, but switch to Trace only in the future to avoid confusion and improve type safety/consistency.
        trace: Union[Trace, str],
        criteria: Optional[str] = None,
        examples: Optional[List[Tuple[Dict[str, str], int]]] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Evaluate the quality of an agentic execution trace using a rubric focused on workflow efficiency toward the user goal.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_workflow_efficiency = (
                Feedback(provider.trajectory_workflow_efficiency_with_cot_reasons)
                .on({
                    "trace": Selector(trace_level=True),
                })
            ```

        Args:
            trace (Union[Trace, str]): The execution trace to evaluate (e.g., as a JSON string or formatted log).
            criteria (Optional[str]): Optional custom criteria for evaluation. Defaults to None.
            examples (Optional[List[Tuple[Dict[str, str], int]]): Optional few-shot examples for evaluation. Defaults to None.
            min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
            max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.
        Returns:
            Tuple[float, Dict]: A tuple containing a value between 0.0 (highly inefficient workflow) and 1.0 (highly streamlined/optimized workflow) and a dictionary containing the reasons for the evaluation.
        """
        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        system_prompt = (
            feedback_v2.TrajectoryWorkflowEfficiency.generate_system_prompt(
                min_score=min_score_val,
                max_score=max_score_val,
                criteria=criteria,
                output_space=output_space,
                examples=examples,
            )
        )

        if isinstance(trace, Trace):
            trajectory = trace.events.to_json()
        elif isinstance(trace, str):
            trajectory = trace
        else:
            raise ValueError(
                f"Invalid trace type: {type(trace)}. Must be a Trace or a string."
            )

        user_prompt = (
            feedback_v2.TrajectoryWorkflowEfficiency.user_prompt.format(
                trajectory=trajectory
            )
        )

        user_prompt = user_prompt.replace(
            "WORKFLOW EFFICIENCY SCORE:", feedback_prompts.COT_REASONS_TEMPLATE
        )

        return self.generate_score_and_reasons(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

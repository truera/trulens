from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import logging
import re
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple
import warnings

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from trulens.core.feedback.provider import Provider
from trulens.feedback import prompts
from trulens.feedback.generated import re_configured_rating
from trulens.feedback.v2.feedback import ContextRelevance
from trulens.feedback.v2.feedback import OutputSpace

logger = logging.getLogger(__name__)


class LLMProvider(Provider):
    """An LLM-based provider.

    This is an abstract class and needs to be initialized as one of these:

    * [OpenAI][trulens.providers.openai.OpenAI] and subclass
      [AzureOpenAI][trulens.providers.openai.AzureOpenAI].

    * [Bedrock][trulens.providers.bedrock.Bedrock].

    * [LiteLLM][trulens.providers.litellm.LiteLLM]. LiteLLM provides an
    interface to a [wide range of
    models](https://docs.litellm.ai/docs/providers).

    * [Langchain][trulens.providers.langchain.Langchain].

    """

    # NOTE(piotrm): "model_" prefix for attributes is "protected" by pydantic v2
    # by default. Need the below adjustment but this means we don't get any
    # warnings if we try to override some internal pydantic name.
    model_engine: str

    model_config: ClassVar[dict] = dict(protected_namespaces=())

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
        **kwargs,
    ) -> str:
        """
        Chat Completion Model

        Returns:
            str: Completion model response.
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
            system_prompt: A pre-formatted system prompt.

            user_prompt: An optional user prompt.

            min_score_val: The minimum score value.

            max_score_val: The maximum score value.

            temperature: The temperature for the LLM response.

        Returns:
            The score on a 0-1 scale.
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
        )

        return (
            re_configured_rating(
                response,
                min_score_val=min_score_val,
                max_score_val=max_score_val,
            )
            - min_score_val
        ) / (max_score_val - min_score_val)

    def generate_confidence_score(
        self,
        verb_confidence_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 10,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Base method to generate a score normalized to 0 to 1, used for evaluation.

        Args:
            system_prompt: A pre-formatted system prompt.

            user_prompt: An optional user prompt.

            min_score_val: The minimum score value.

            max_score_val: The maximum score value.

            temperature: The temperature for the LLM response.

        Returns:
            The feedback score on a 0-1 scale and the confidence score.
        """
        assert self.endpoint is not None, "Endpoint is not set."
        assert (
            max_score_val > min_score_val
        ), "Max score must be greater than min score."

        llm_messages = [{"role": "system", "content": verb_confidence_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})

        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            messages=llm_messages,
            temperature=temperature,
        )
        # print(f'Relevance score and Confidence score: {response}')
        relevance_score = re.search(r"\d+", response)

        confidence_score = re.search(
            r"CONFIDENCE: (\d+)", response
        )  # TODO: refactor this to use a more general pattern
        if (
            confidence_score
            and relevance_score
            and 0 <= float(confidence_score.group(1)) <= 1
            and min_score_val
            <= float(relevance_score.group(0))
            <= max_score_val
        ):
            confidence = float(confidence_score.group(1))
            relevance_score = float(relevance_score.group(0))

            return (
                (relevance_score - min_score_val)
                / (max_score_val - min_score_val),
                {"confidence_score": confidence},
            )
        else:
            raise ValueError("Confidence score not found in response.")

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
            system_prompt: A pre-formatted system prompt.

            user_prompt: An optional user prompt. Defaults to None.

            min_score_val: The minimum score value.

            max_score_val: The maximum score value.

            temperature: The temperature for the LLM response.

        Returns:
            The score on a 0-1 scale.

            Reason metadata if returned by the LLM.
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
        )
        if "Supporting Evidence" in response:
            score = -1
            supporting_evidence = None
            criteria = None
            for line in response.split("\n"):
                if "Score" in line:
                    score = (
                        re_configured_rating(
                            line,
                            min_score_val=min_score_val,
                            max_score_val=max_score_val,
                        )
                        - min_score_val
                    ) / (max_score_val - min_score_val)
                criteria_lines = []
                supporting_evidence_lines = []
                collecting_criteria = False
                collecting_evidence = False

                for line in response.split("\n"):
                    if "Criteria:" in line:
                        criteria_lines.append(
                            line.split("Criteria:", 1)[1].strip()
                        )
                        collecting_criteria = True
                        collecting_evidence = False
                    elif "Supporting Evidence:" in line:
                        supporting_evidence_lines.append(
                            line.split("Supporting Evidence:", 1)[1].strip()
                        )
                        collecting_evidence = True
                        collecting_criteria = False
                    elif collecting_criteria:
                        if "Supporting Evidence:" not in line:
                            criteria_lines.append(line.strip())
                        else:
                            collecting_criteria = False
                    elif collecting_evidence:
                        if "Criteria:" not in line:
                            supporting_evidence_lines.append(line.strip())
                        else:
                            collecting_evidence = False

                criteria = "\n".join(criteria_lines).strip()
                supporting_evidence = "\n".join(
                    supporting_evidence_lines
                ).strip()
            reasons = {
                "reason": (
                    f"{'Criteria: ' + str(criteria)}\n"
                    f"{'Supporting Evidence: ' + str(supporting_evidence)}"
                )
            }
            return score, reasons

        else:
            score = (
                re_configured_rating(
                    response,
                    min_score_val=min_score_val,
                    max_score_val=max_score_val,
                )
                - min_score_val
            ) / (max_score_val - min_score_val)
            warnings.warn(
                "No supporting evidence provided. Returning score only.",
                UserWarning,
            )
            return score, {}

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
        for output in OutputSpace:
            if output.value == (min_score_val, max_score_val):
                return output.name
        raise ValueError(
            f"Invalid score range: [{min_score_val}, {max_score_val}]. Must match one of the predefined output spaces."
        )

    def context_relevance(
        self,
        question: str,
        context: str,
        criteria: str = "",
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the relevance of the context to the question.

        Example:

            ```python
            from trulens.instrument.langchain import TruChain
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
            criteria (str): Overriding evaluation criteria for evaluation .
            min_score_val (int): The minimum score value. Defaults to 0.
            max_score_val (int): The maximum score value. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.
        Returns:
            float: A value between 0.0 (not relevant) and 1.0 (relevant).
        """
        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        if criteria and output_space:
            ContextRelevance.override_critera_and_output_space(
                criteria, output_space
            )

        return self.generate_score(
            system_prompt=ContextRelevance.system_prompt,
            user_prompt=str.format(
                prompts.CONTEXT_RELEVANCE_USER,
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
        criteria: str = "",
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
            from trulens.instrument.langchain import TruChain
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
            criteria (str): Overriding evaluation criteria for evaluation .
            min_score_val (int): The minimum score value. Defaults to 0.
            max_score_val (int): The maximum score value. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being "relevant".
        """

        user_prompt = str.format(
            prompts.CONTEXT_RELEVANCE_USER, question=question, context=context
        )
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        if criteria and output_space:
            ContextRelevance.override_critera_and_output_space(
                criteria, output_space
            )

        return self.generate_score_and_reasons(
            system_prompt=ContextRelevance.system_prompt,
            user_prompt=user_prompt,
            min_score_val=min_score_val,
            max_score_val=max_score_val,
            temperature=temperature,
        )

    def context_relevance_verb_confidence(
        self,
        question: str,
        context: str,
        criteria: str = "",
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Uses chat completion model. A function that completes a
        template to check the relevance of the context to the question.
        Also uses chain of thought methodology and emits the reasons.

        Example:

            ```python
            from trulens.instrument.llamaindex import TruLlama
            context = TruLlama.select_context(llamaindex_rag_app)
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
            criteria (str): Overriding evaluation criteria for evaluation .
            min_score_val (int): The minimum score value. Defaults to 0.
            max_score_val (int): The maximum score value. Defaults to 3.
            temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.
        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being "relevant".
            Dict[str, float]: A dictionary containing the confidence score.
        """

        output_space = self._determine_output_space(
            min_score_val, max_score_val
        )

        if criteria and output_space:
            ContextRelevance.override_critera_and_output_space(
                criteria, output_space
            )

        try:
            return self.generate_confidence_score(
                verb_confidence_prompt=ContextRelevance.system_prompt
                + ContextRelevance.verb_confidence_prompt,
                user_prompt=str.format(
                    prompts.CONTEXT_RELEVANCE_USER,
                    question=question,
                    context=context,
                ),
                min_score_val=min_score_val,
                max_score_val=max_score_val,
                temperature=temperature,
            )
        except ValueError as e:
            logger.error(e)
            return None, None

    def relevance(self, prompt: str, response: str) -> float:
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

        Parameters:
            prompt (str): A text prompt to an agent.

            response (str): The agent's response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
                "relevant".
        """
        return self.generate_score(
            system_prompt=prompts.ANSWER_RELEVANCE_SYSTEM,
            user_prompt=str.format(
                prompts.ANSWER_RELEVANCE_USER, prompt=prompt, response=response
            ),
        )

    def relevance_with_cot_reasons(
        self, prompt: str, response: str
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

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
                "relevant".
        """
        system_prompt = prompts.ANSWER_RELEVANCE_SYSTEM

        user_prompt = str.format(
            prompts.ANSWER_RELEVANCE_USER, prompt=prompt, response=response
        )
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )
        return self.generate_score_and_reasons(system_prompt, user_prompt)

    def sentiment(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the sentiment of some text.

        Example:

            ```python
            feedback = Feedback(provider.sentiment).on_output()
            ```

        Args:
            text: The text to evaluate sentiment of.

        Returns:
            A value between 0 and 1. 0 being "negative sentiment" and 1
                being "positive sentiment".
        """
        system_prompt = prompts.SENTIMENT_SYSTEM
        user_prompt = prompts.SENTIMENT_USER + text
        return self.generate_score(system_prompt, user_prompt)

    def sentiment_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
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

        Returns:
            float: A value between 0.0 (negative sentiment) and 1.0 (positive sentiment).
        """
        system_prompt = prompts.SENTIMENT_SYSTEM
        user_prompt = (
            prompts.SENTIMENT_USER + text + prompts.COT_REASONS_TEMPLATE
        )
        return self.generate_score_and_reasons(system_prompt, user_prompt)

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
            prompt=prompts.CORRECT_SYSTEM
        )
        agreement_txt = self._get_answer_agreement(
            prompt, response, chat_response
        )
        return (
            re_configured_rating(
                agreement_txt, min_score_val=0, max_score_val=3
            )
            / 3
        )

    def _langchain_evaluate(self, text: str, criteria: str) -> float:
        """
        Uses chat completion model. A general function that completes a template
        to evaluate different aspects of some text. Prompt credit to Langchain.

        Args:
            text (str): A prompt to an agent.
            criteria (str): The specific criteria for evaluation.

        Returns:
            float: A value between 0.0 and 1.0, representing the specified
                evaluation.
        """

        system_prompt = str.format(
            prompts.LANGCHAIN_PROMPT_TEMPLATE_SYSTEM, criteria=criteria
        )
        user_prompt = str.format(
            prompts.LANGCHAIN_PROMPT_TEMPLATE_USER, submission=text
        )

        return self.generate_score(system_prompt, user_prompt)

    def _langchain_evaluate_with_cot_reasons(
        self, text: str, criteria: str
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A general function that completes a template
        to evaluate different aspects of some text. Prompt credit to Langchain.

        Args:
            text (str): A prompt to an agent.
            criteria (str): The specific criteria for evaluation.

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 and 1.0, representing the specified evaluation, and a string containing the reasons for the evaluation.
        """

        system_prompt = str.format(
            prompts.LANGCHAIN_PROMPT_TEMPLATE_WITH_COT_REASONS_SYSTEM,
            criteria=criteria,
        )
        user_prompt = str.format(
            prompts.LANGCHAIN_PROMPT_TEMPLATE_USER, submission=text
        )
        return self.generate_score_and_reasons(system_prompt, user_prompt)

    def conciseness(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the conciseness of some text. Prompt credit to LangChain Eval.

        Example:

            ```python
            feedback = Feedback(provider.conciseness).on_output()
            ```

        Args:
            text: The text to evaluate the conciseness of.

        Returns:
            A value between 0.0 (not concise) and 1.0 (concise).

        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_CONCISENESS_SYSTEM_PROMPT
        )

    def conciseness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the conciseness of some text. Prompt credit to LangChain Eval.

        Example:

            ```python
            feedback = Feedback(provider.conciseness).on_output()
            ```
        Args:
            text: The text to evaluate the conciseness of.

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 (not concise) and 1.0 (concise) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_CONCISENESS_SYSTEM_PROMPT
        )

    def correctness(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the correctness of some text. Prompt credit to LangChain Eval.

        Example:

            ```python
            feedback = Feedback(provider.correctness).on_output()
            ```

        Args:
            text: A prompt to an agent.

        Returns:
            A value between 0.0 (not correct) and 1.0 (correct).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_CORRECTNESS_SYSTEM_PROMPT
        )

    def correctness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0 (not correct) and 1.0 (correct) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_CORRECTNESS_SYSTEM_PROMPT
        )

    def coherence(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a
        template to check the coherence of some text. Prompt credit to LangChain Eval.

        Example:

            ```python
            feedback = Feedback(provider.coherence).on_output()
            ```

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not coherent) and 1.0 (coherent).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_COHERENCE_SYSTEM_PROMPT
        )

    def coherence_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0 (not coherent) and 1.0 (coherent) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_COHERENCE_SYSTEM_PROMPT
        )

    def harmfulness(self, text: str) -> float:
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
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_HARMFULNESS_SYSTEM_PROMPT
        )

    def harmfulness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0 (not harmful) and 1.0 (harmful) and a string containing the reasons for the evaluation.
        """

        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_HARMFULNESS_SYSTEM_PROMPT
        )

    def maliciousness(self, text: str) -> float:
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

        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_MALICIOUSNESS_SYSTEM_PROMPT
        )

    def maliciousness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat compoletion model. A function that completes a
        template to check the maliciousness of some text. Prompt credit to LangChain Eval.
        Also uses chain of thought methodology and emits the reasons.

        Example:

            ```python
            feedback = Feedback(provider.maliciousness_with_cot_reasons).on_output()
            ```

        Args:
            text (str): The text to evaluate.

        Returns:
            Tuple[float, str]: A tuple containing a value between 0 (not malicious) and 1.0 (malicious) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_MALICIOUSNESS_SYSTEM_PROMPT
        )

    def helpfulness(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the helpfulness of some text. Prompt credit to LangChain Eval.

        Example:

            ```python
            feedback = Feedback(provider.helpfulness).on_output()
            ```

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not helpful) and 1.0 (helpful).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_HELPFULNESS_SYSTEM_PROMPT
        )

    def helpfulness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0 (not helpful) and 1.0 (helpful) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_HELPFULNESS_SYSTEM_PROMPT
        )

    def controversiality(self, text: str) -> float:
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

        Returns:
            float: A value between 0.0 (not controversial) and 1.0
                (controversial).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_CONTROVERSIALITY_SYSTEM_PROMPT
        )

    def controversiality_with_cot_reasons(
        self, text: str
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0 (not controversial) and 1.0 (controversial) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_CONTROVERSIALITY_SYSTEM_PROMPT
        )

    def misogyny(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the misogyny of some text. Prompt credit to LangChain Eval.

        Example:

            ```python
            feedback = Feedback(provider.misogyny).on_output()
            ```

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not misogynistic) and 1.0 (misogynistic).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_MISOGYNY_SYSTEM_PROMPT
        )

    def misogyny_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 (not misogynistic) and 1.0 (misogynistic) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_MISOGYNY_SYSTEM_PROMPT
        )

    def criminality(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the criminality of some text. Prompt credit to LangChain Eval.

        Example:

            ```python
            feedback = Feedback(provider.criminality).on_output()
            ```

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not criminal) and 1.0 (criminal).

        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_CRIMINALITY_SYSTEM_PROMPT
        )

    def criminality_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 (not criminal) and 1.0 (criminal) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_CRIMINALITY_SYSTEM_PROMPT
        )

    def insensitivity(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the insensitivity of some text. Prompt credit to LangChain Eval.

        Example:

            ```python
            feedback = Feedback(provider.insensitivity).on_output()
            ```

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not insensitive) and 1.0 (insensitive).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_INSENSITIVITY_SYSTEM_PROMPT
        )

    def insensitivity_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 (not insensitive) and 1.0 (insensitive) and a string containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_INSENSITIVITY_SYSTEM_PROMPT
        )

    def _get_answer_agreement(
        self, prompt: str, response: str, check_response: str
    ) -> str:
        """
        Uses chat completion model. A function that completes a template to
        check if two answers agree.

        Args:
            text (str): A prompt to an agent.
            response (str): The agent's response to the prompt.
            check_response(str): The response to check against.

        Returns:
            str
        """

        assert self.endpoint is not None, "Endpoint is not set."

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            prompt=(prompts.AGREEMENT_SYSTEM % (prompt, check_response))
            + response,
        )

    def _generate_key_points(self, source: str):
        """
        Uses chat completion model. A function that tries to distill main points
        to be used by the comprehensiveness feedback function.

         Args:
            source (str): Text corresponding to source material.

        Returns:
            (str) key points of the source text.
        """

        return self._create_chat_completion(
            prompt=prompts.GENERATE_KEY_POINTS_SYSTEM_PROMPT
            + str.format(prompts.GENERATE_KEY_POINTS_USER_PROMPT, source=source)
        )

    def _assess_key_point_inclusion(
        self, key_points: str, summary: str
    ) -> List:
        """
        Splits key points by newlines and assesses if each one is included in the summary.

        Args:
            key_points (str): Key points separated by newlines.
            summary (str): The summary text to check for inclusion of key points.

        Returns:
            List[str]: A list of strings indicating whether each key point is included in the summary.
        """
        key_points_list = key_points.split("\n")

        system_prompt = prompts.COMPREHENSIVENESS_SYSTEM_PROMPT
        inclusion_assessments = []
        for key_point in key_points_list:
            user_prompt = str.format(
                prompts.COMPOREHENSIVENESS_USER_PROMPT,
                key_point=key_point,
                summary=summary,
            )
            inclusion_assessment = self._create_chat_completion(
                prompt=system_prompt + user_prompt
            )
            inclusion_assessments.append(inclusion_assessment)

        return inclusion_assessments

    def comprehensiveness_with_cot_reasons(
        self, source: str, summary: str
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

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 (not comprehensive) and 1.0 (comprehensive) and a string containing the reasons for the evaluation.
        """

        key_points = self._generate_key_points(source)
        key_point_inclusion_assessments = self._assess_key_point_inclusion(
            key_points, summary
        )
        scores = []
        reasons = ""
        for assessment in key_point_inclusion_assessments:
            reasons += assessment + "\n\n"
            if assessment:
                first_line = assessment.split("\n")[0]
                score = (
                    re_configured_rating(
                        first_line, min_score_val=0, max_score_val=3
                    )
                    / 3
                )
                scores.append(score)

        score = sum(scores) / len(scores) if scores else 0
        return score, {"reasons": reasons}

    def summarization_with_cot_reasons(
        self, source: str, summary: str
    ) -> Tuple[float, Dict]:
        """
        Summarization is deprecated in place of comprehensiveness. This function is no longer implemented.
        """
        raise NotImplementedError(
            "summarization_with_cot_reasons is deprecated and not implemented. Please use comprehensiveness_with_cot_reasons instead."
        )

    def stereotypes(self, prompt: str, response: str) -> float:
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

        Returns:
            A value between 0.0 (no stereotypes assumed) and 1.0 (stereotypes assumed).
        """
        system_prompt = prompts.STEREOTYPES_SYSTEM_PROMPT
        user_prompt = str.format(
            prompts.STEREOTYPES_USER_PROMPT, prompt=prompt, response=response
        )
        return self.generate_score(system_prompt, user_prompt)

    def stereotypes_with_cot_reasons(
        self, prompt: str, response: str
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check adding assumed stereotypes in the response when not present in the
        prompt.

        Example:

            ```python
            feedback = Feedback(provider.stereotypes_with_cot_reasons).on_input_output()
            ```

        Args:
            prompt (str): A text prompt to an agent.

            response (str): The agent's response to the prompt.

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 (no stereotypes assumed) and 1.0 (stereotypes assumed) and a string containing the reasons for the evaluation.
        """
        system_prompt = (
            prompts.STEREOTYPES_SYSTEM_PROMPT + prompts.COT_REASONS_TEMPLATE
        )
        user_prompt = str.format(
            prompts.STEREOTYPES_USER_PROMPT, prompt=prompt, response=response
        )

        return self.generate_score_and_reasons(system_prompt, user_prompt)

    def groundedness_measure_with_cot_reasons(
        self, source: str, statement: str
    ) -> Tuple[float, dict]:
        """A measure to track if the source material supports each sentence in
        the statement using an LLM provider.

        The LLM will process the entire statement at once, using chain of
        thought methodology to emit the reasons.

        Abstentions will be considered as grounded.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons)
                .on(context.collect()
                .on_output()
                )
            ```
        Args:
            source: The source that should support the statement.
            statement: The statement to check groundedness.

        Returns:
            Tuple[float, dict]: A tuple containing a value between 0.0 (not grounded) and 1.0 (grounded) and a dictionary containing the reasons for the evaluation.
        """
        nltk.download("punkt", quiet=True)
        groundedness_scores = {}
        reasons_str = ""

        hypotheses = sent_tokenize(statement)

        system_prompt = prompts.LLM_GROUNDEDNESS_SYSTEM

        def evaluate_hypothesis(index, hypothesis):
            user_prompt = prompts.LLM_GROUNDEDNESS_USER.format(
                premise=f"{source}", hypothesis=f"{hypothesis}"
            )
            score, reason = self.generate_score_and_reasons(
                system_prompt, user_prompt
            )
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

    def groundedness_measure_with_cot_reasons_consider_answerability(
        self, source: str, statement: str, question: str
    ) -> Tuple[float, dict]:
        """A measure to track if the source material supports each sentence in
        the statement using an LLM provider.

        The LLM will process the entire statement at once, using chain of
        thought methodology to emit the reasons.

        In the case of abstentions, the LLM will be asked to consider the answerability of the question given the source material.

        If the question is considered answerable, abstentions will be considered as not grounded and punished with low scores. Otherwise, unanswerable abstentions will be considered grounded.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons)
                .on(context.collect()
                .on_output()
                .on_input()
                )
            ```
        Args:
            source: The source that should support the statement.
            statement: The statement to check groundedness.
            question: The question to check answerability.

        Returns:
            Tuple[float, dict]: A tuple containing a value between 0.0 (not grounded) and 1.0 (grounded) and a dictionary containing the reasons for the evaluation.
        """
        nltk.download("punkt", quiet=True)
        groundedness_scores = {}
        reasons_str = ""

        def evaluate_abstention(statement):
            user_prompt = prompts.LLM_ABSTENTION_USER.format(
                statement=statement
            )
            score = self.generate_score(
                prompts.LLM_ABSTENTION_SYSTEM, user_prompt
            )
            return score

        def evaluate_answerability(question, source):
            user_prompt = prompts.LLM_ANSWERABILITY_USER.format(
                question=question, source=source
            )
            score = self.generate_score(
                prompts.LLM_ANSWERABILITY_SYSTEM, user_prompt
            )
            return score

        hypotheses = sent_tokenize(statement)

        system_prompt = prompts.LLM_GROUNDEDNESS_SYSTEM

        def evaluate_hypothesis(index, hypothesis):
            abstention_score = evaluate_abstention(hypothesis)
            if abstention_score > 0.5:
                answerability_score = evaluate_answerability(question, source)
                if answerability_score > 0.5:
                    return index, 0.0, {"reason": "Answerable abstention"}
                else:
                    return index, 1.0, {"reason": "Unanswerable abstention"}
            else:
                user_prompt = prompts.LLM_GROUNDEDNESS_USER.format(
                    premise=f"{source}", hypothesis=f"{hypothesis}"
                )
                score, reason = self.generate_score_and_reasons(
                    system_prompt, user_prompt
                )
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

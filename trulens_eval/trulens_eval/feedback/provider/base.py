import logging
from typing import ClassVar, Dict, Optional, Sequence, Tuple
import warnings

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.generated import re_0_10_rating
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel

logger = logging.getLogger(__name__)


class Provider(WithClassInfo, SerialModel):
    """Base Provider class."""

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    endpoint: Optional[Endpoint] = None
    """Endpoint supporting this provider.
    
    Remote API invocations are handled by the endpoint.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)


class LLMProvider(Provider):
    """An LLM-based provider.
    
    This is an abstract class and needs to be initialized as one of these:

    * [OpenAI][trulens_eval.feedback.provider.openai.OpenAI] and subclass
      [AzureOpenAI][trulens_eval.feedback.provider.openai.AzureOpenAI].

    * [Bedrock][trulens_eval.feedback.provider.bedrock.Bedrock].

    * [LiteLLM][trulens_eval.feedback.provider.litellm.LiteLLM]. LiteLLM provides an
    interface to a [wide range of
    models](https://docs.litellm.ai/docs/providers).
    
    * [Langchain][trulens_eval.feedback.provider.langchain.Langchain].

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

    #@abstractmethod
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        """
        Chat Completion Model

        Returns:
            str: Completion model response.
        """
        # text
        raise NotImplementedError()
        pass

    def _find_relevant_string(self, full_source: str, hypothesis: str) -> str:
        assert self.endpoint is not None, "Endpoint is not set."

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            prompt=str.format(
                prompts.SYSTEM_FIND_SUPPORTING,
                prompt=full_source,
            ) + "\n" +
            str.format(prompts.USER_FIND_SUPPORTING, response=hypothesis)
        )

    def _summarized_groundedness(self, premise: str, hypothesis: str) -> float:
        """
        A groundedness measure best used for summarized premise against simple
        hypothesis. This LLM implementation uses information overlap prompts.

        Args:
            premise (str): Summarized source sentences.
            hypothesis (str): Single statement setnece.

        Returns:
            float: Information Overlap
        """
        return self.generate_score(
            system_prompt=str.format(
                prompts.LLM_GROUNDEDNESS,
                premise=premise,
                hypothesis=hypothesis,
            )
        )

    def _groundedness_doc_in_out(self, premise: str, hypothesis: str) -> str:
        """
        An LLM prompt using the entire document for premise and entire statement
        document for hypothesis.

        Args:
            premise (str): A source document
            hypothesis (str): A statement to check

        Returns:
            str: An LLM response using a scorecard template
        """
        assert self.endpoint is not None, "Endpoint is not set."

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            prompt=str.format(prompts.LLM_GROUNDEDNESS_FULL_SYSTEM,) +
            str.format(
                prompts.LLM_GROUNDEDNESS_FULL_PROMPT,
                premise=premise,
                hypothesis=hypothesis
            )
        )

    def generate_score(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        normalize: float = 10.0
    ) -> float:
        """
        Base method to generate a score only, used for evaluation.

        Args:
            system_prompt (str): A pre-formated system prompt

        Returns:
            The score (float): 0-1 scale.
        """
        assert self.endpoint is not None, "Endpoint is not set."

        llm_messages = [{"role": "system", "content": system_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})

        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion, messages=llm_messages
        )

        return re_0_10_rating(response) / normalize

    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        normalize: float = 10.0
    ) -> Tuple[float, Dict]:
        """
        Base method to generate a score and reason, used for evaluation.

        Args:
            system_prompt (str): A pre-formated system prompt

        Returns:
            The score (float): 0-1 scale and reason metadata (dict) if returned by the LLM.
        """
        assert self.endpoint is not None, "Endpoint is not set."

        llm_messages = [{"role": "system", "content": system_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})

        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion, messages=llm_messages
        )
        if "Supporting Evidence" in response:
            score = -1
            supporting_evidence = None
            criteria = None
            for line in response.split('\n'):
                if "Score" in line:
                    score = re_0_10_rating(line) / normalize
                criteria_lines = []
                supporting_evidence_lines = []
                collecting_criteria = False
                collecting_evidence = False

                for line in response.split('\n'):
                    if "Criteria:" in line:
                        criteria_lines.append(line.split("Criteria:", 1)[1].strip())
                        collecting_criteria = True
                        collecting_evidence = False
                    elif "Supporting Evidence:" in line:
                        supporting_evidence_lines.append(line.split("Supporting Evidence:", 1)[1].strip())
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
                supporting_evidence = "\n".join(supporting_evidence_lines).strip()
            reasons = {
                'reason':
                    (
                        f"{'Criteria: ' + str(criteria)}\n"
                        f"{'Supporting Evidence: ' + str(supporting_evidence)}"
                    )
            }
            return score, reasons

        else:
            score = re_0_10_rating(response) / normalize
            warnings.warn(
                "No supporting evidence provided. Returning score only.",
                UserWarning
            )
            return score, {}

    def qs_relevance(self, question: str, statement: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the relevance of the statement to the question.

        ```python
        feedback = Feedback(provider.qs_relevance).on_input_output() 
        ```
        The `on_input_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)
        
        Usage on RAG Contexts:

        ```python
        feedback = Feedback(provider.qs_relevance).on_input().on(
            TruLlama.select_source_nodes().node.text # See note below
        ).aggregate(np.mean) 
        ```

        The `on(...)` selector can be changed. See [Feedback Function Guide :
        Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

        Args:
            question (str): A question being asked. 
            statement (str): A statement to the question.

        Returns:
            float: A value between 0.0 (not relevant) and 1.0 (relevant).
        """
        return self.generate_score(
            system_prompt=str.format(
                prompts.QS_RELEVANCE, question=question, statement=statement
            )
        )

    def qs_relevance_with_cot_reasons(self, question: str,
                                      statement: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a
        template to check the relevance of the statement to the question.
        Also uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```
        feedback = Feedback(provider.qs_relevance_with_cot_reasons).on_input_output() 
        ```
        The `on_input_output()` selector can be changed. See [Feedback Function Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)
        
        Usage on RAG Contexts:
        ```
        feedback = Feedback(provider.qs_relevance_with_cot_reasons).on_input().on(
            TruLlama.select_source_nodes().node.text # See note below
        ).aggregate(np.mean) 

        ```
        The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)


        Args:
            question (str): A question being asked. 
            statement (str): A statement to the question.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being "relevant".
        """
        system_prompt = str.format(
            prompts.QS_RELEVANCE, question=question, statement=statement
        )
        system_prompt = system_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )
        return self.generate_score_and_reasons(system_prompt)

    def relevance(self, prompt: str, response: str) -> float:
        """
        Uses chat completion model. A function that completes a
        template to check the relevance of the response to a prompt.

        **Usage:**
        ```python
        feedback = Feedback(provider.relevance).on_input_output()
        ```
        
        The `on_input_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Usage on RAG Contexts:

        ```python
        feedback = Feedback(provider.relevance).on_input().on(
            TruLlama.select_source_nodes().node.text # See note below
        ).aggregate(np.mean) 
        ```

        The `on(...)` selector can be changed. See [Feedback Function Guide :
        Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

        Parameters:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return self.generate_score(
            system_prompt=str.
            format(prompts.PR_RELEVANCE, prompt=prompt, response=response)
        )

    def relevance_with_cot_reasons(self, prompt: str,
                                   response: str) -> Tuple[float, Dict]:
        """
        Uses chat completion Model. A function that completes a template to
        check the relevance of the response to a prompt. Also uses chain of
        thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.relevance_with_cot_reasons).on_input_output()
        ```

        The `on_input_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Usage on RAG Contexts:
        ```python

        feedback = Feedback(provider.relevance_with_cot_reasons).on_input().on(
            TruLlama.select_source_nodes().node.text # See note below
        ).aggregate(np.mean) 
        ```

        The `on(...)` selector can be changed. See [Feedback Function Guide :
        Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

        Args:
            prompt (str): A text prompt to an agent. 
            response (str): The agent's response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        system_prompt = str.format(
            prompts.PR_RELEVANCE, prompt=prompt, response=response
        )
        system_prompt = system_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )
        return self.generate_score_and_reasons(system_prompt)

    def sentiment(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the sentiment of some text.

        Usage:
            ```python
            feedback = Feedback(provider.sentiment).on_output() 
            ```

            The `on_output()` selector can be changed. See [Feedback Function
            Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text: The text to evaluate sentiment of.

        Returns:
            A value between 0 and 1. 0 being "negative sentiment" and 1
                being "positive sentiment".
        """
        system_prompt = prompts.SENTIMENT_SYSTEM_PROMPT + text
        return self.generate_score(system_prompt=system_prompt)

    def sentiment_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a
        template to check the sentiment of some text.
        Also uses chain of thought methodology and emits the reasons.

        **Usage:**

        ```python
        feedback = Feedback(provider.sentiment_with_cot_reasons).on_output() 
        ```
        
        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (negative sentiment) and 1.0 (positive sentiment).
        """
        system_prompt = prompts.SENTIMENT_SYSTEM_PROMPT
        system_prompt = system_prompt + prompts.COT_REASONS_TEMPLATE
        return self.generate_score_and_reasons(system_prompt, user_prompt=text)

    def model_agreement(self, prompt: str, response: str) -> float:
        """
        Uses chat completion model. A function that gives a chat completion model the same
        prompt and gets a response, encouraging truthfulness. A second template
        is given to the model with a prompt that the original response is
        correct, and measures whether previous chat completion response is similar.

        **Usage:**

        ```python
        feedback = Feedback(provider.model_agreement).on_input_output() 
        ```

        The `on_input_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Parameters:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.

        Returns:
            float: A value between 0.0 (not in agreement) and 1.0 (in agreement).
        """
        warnings.warn(
            "`model_agreement` has been deprecated. "
            "Use `GroundTruthAgreement(ground_truth)` instead.",
            DeprecationWarning
        )
        chat_response = self._create_chat_completion(
            prompt=prompts.CORRECT_SYSTEM_PROMPT
        )
        agreement_txt = self._get_answer_agreement(
            prompt, response, chat_response
        )
        return re_0_10_rating(agreement_txt) / 10.0

    def _langchain_evaluate(self, text: str, criteria: str) -> float:
        """
        Uses chat completion model. A general function that completes a template
        to evaluate different aspects of some text. Prompt credit to Langchain.

        Parameters:
            text (str): A prompt to an agent.
            criteria (str): The specific criteria for evaluation.

        Returns:
            float: A value between 0.0 and 1.0, representing the specified
            evaluation.
        """

        system_prompt = str.format(
            prompts.LANGCHAIN_PROMPT_TEMPLATE,
            criteria=criteria,
            submission=text
        )

        return self.generate_score(system_prompt=system_prompt)

    def _langchain_evaluate_with_cot_reasons(
        self, text: str, criteria: str
    ) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A general function that completes a template
        to evaluate different aspects of some text. Prompt credit to Langchain.

        Parameters:
            text (str): A prompt to an agent.
            criteria (str): The specific criteria for evaluation.

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 and 1.0, representing the specified
            evaluation, and a string containing the reasons for the evaluation.
        """

        system_prompt = str.format(
            prompts.LANGCHAIN_PROMPT_TEMPLATE_WITH_COT_REASONS,
            criteria=criteria,
            submission=text
        )
        return self.generate_score_and_reasons(system_prompt=system_prompt)

    def conciseness(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the conciseness of some text. Prompt credit to Langchain Eval.

        Usage:
            ```python
            feedback = Feedback(provider.conciseness).on_output() 
            ```

            The `on_output()` selector can be changed. See [Feedback Function
            Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text: The text to evaluate the conciseness of.

        Returns:
            A value between 0.0 (not concise) and 1.0 (concise).

        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_CONCISENESS_PROMPT
        )

    def conciseness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the conciseness of some text. Prompt credit to Langchain Eval.

        Usage:
            ```python
            feedback = Feedback(provider.conciseness).on_output() 
            ```

            The `on_output()` selector can be changed. See [Feedback Function
            Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text: The text to evaluate the conciseness of.

        Returns:
            A value between 0.0 (not concise) and 1.0 (concise)
            
            A dictionary containing the reasons for the evaluation.
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_CONCISENESS_PROMPT
        )

    def correctness(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the correctness of some text. Prompt credit to Langchain Eval.

        Usage:
            ```python
            feedback = Feedback(provider.correctness).on_output() 
            ```

            The `on_output()` selector can be changed. See [Feedback Function
            Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Parameters:
            text: A prompt to an agent.

        Returns:
            A value between 0.0 (not correct) and 1.0 (correct).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_CORRECTNESS_PROMPT
        )

    def correctness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the correctness of some text. Prompt credit to Langchain Eval.
        Also uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.correctness_with_cot_reasons).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not correct) and 1.0 (correct).
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_CORRECTNESS_PROMPT
        )

    def coherence(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a
        template to check the coherence of some text. Prompt credit to Langchain Eval.
        
        **Usage:**
        ```python
        feedback = Feedback(provider.coherence).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not coherent) and 1.0 (coherent).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_COHERENCE_PROMPT
        )

    def coherence_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the coherence of some text. Prompt credit to Langchain Eval. Also
        uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.coherence_with_cot_reasons).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not coherent) and 1.0 (coherent).
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_COHERENCE_PROMPT
        )

    def harmfulness(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the harmfulness of some text. Prompt credit to Langchain Eval.

        **Usage:**
        ```python
        feedback = Feedback(provider.harmfulness).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)
        
        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not harmful) and 1.0 (harmful)".
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_HARMFULNESS_PROMPT
        )

    def harmfulness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the harmfulness of some text. Prompt credit to Langchain Eval.
        Also uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.harmfulness_with_cot_reasons).on_output() 
        
        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not harmful) and 1.0 (harmful).
        """

        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_HARMFULNESS_PROMPT
        )

    def maliciousness(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the maliciousness of some text. Prompt credit to Langchain Eval.

        **Usage:**
        ```python
        feedback = Feedback(provider.maliciousness).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not malicious) and 1.0 (malicious).
        """

        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_MALICIOUSNESS_PROMPT
        )

    def maliciousness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat compoletion model. A function that completes a
        template to check the maliciousness of some text. Prompt credit to Langchain Eval.
        Also uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.maliciousness_with_cot_reasons).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not malicious) and 1.0 (malicious).
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_MALICIOUSNESS_PROMPT
        )

    def helpfulness(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the helpfulness of some text. Prompt credit to Langchain Eval.

        **Usage:**
        ```python
        feedback = Feedback(provider.helpfulness).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not helpful) and 1.0 (helpful).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_HELPFULNESS_PROMPT
        )

    def helpfulness_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the helpfulness of some text. Prompt credit to Langchain Eval.
        Also uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.helpfulness_with_cot_reasons).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not helpful) and 1.0 (helpful).
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_HELPFULNESS_PROMPT
        )

    def controversiality(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the controversiality of some text. Prompt credit to Langchain
        Eval.

        **Usage:**
        ```python
        feedback = Feedback(provider.controversiality).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not controversial) and 1.0
            (controversial).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_CONTROVERSIALITY_PROMPT
        )

    def controversiality_with_cot_reasons(self,
                                          text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the controversiality of some text. Prompt credit to Langchain
        Eval. Also uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.controversiality_with_cot_reasons).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)
        
        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not controversial) and 1.0 (controversial).
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_CONTROVERSIALITY_PROMPT
        )

    def misogyny(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the misogyny of some text. Prompt credit to Langchain Eval.

        **Usage:**
        ```python
        feedback = Feedback(provider.misogyny).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not misogynistic) and 1.0 (misogynistic).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_MISOGYNY_PROMPT
        )

    def misogyny_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the misogyny of some text. Prompt credit to Langchain Eval. Also
        uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.misogyny_with_cot_reasons).on_output() 
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not misogynistic) and 1.0 (misogynistic).
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_MISOGYNY_PROMPT
        )

    def criminality(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the criminality of some text. Prompt credit to Langchain Eval.

        **Usage:**
        ```python
        feedback = Feedback(provider.criminality).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not criminal) and 1.0 (criminal).

        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_CRIMINALITY_PROMPT
        )

    def criminality_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the criminality of some text. Prompt credit to Langchain Eval.
        Also uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.criminality_with_cot_reasons).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not criminal) and 1.0 (criminal).
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_CRIMINALITY_PROMPT
        )

    def insensitivity(self, text: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check the insensitivity of some text. Prompt credit to Langchain Eval.

        **Usage:**
        ```python
        feedback = Feedback(provider.insensitivity).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not insensitive) and 1.0 (insensitive).
        """
        return self._langchain_evaluate(
            text=text, criteria=prompts.LANGCHAIN_INSENSITIVITY_PROMPT
        )

    def insensitivity_with_cot_reasons(self, text: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check the insensitivity of some text. Prompt credit to Langchain Eval.
        Also uses chain of thought methodology and emits the reasons.

        **Usage:**
        ```python
        feedback = Feedback(provider.insensitivity_with_cot_reasons).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): The text to evaluate.

        Returns:
            float: A value between 0.0 (not insensitive) and 1.0 (insensitive).
        """
        return self._langchain_evaluate_with_cot_reasons(
            text=text, criteria=prompts.LANGCHAIN_INSENSITIVITY_PROMPT
        )

    def _get_answer_agreement(
        self, prompt: str, response: str, check_response: str
    ) -> str:
        """
        Uses chat completion model. A function that completes a template to
        check if two answers agree.

        Parameters:
            text (str): A prompt to an agent.
            response (str): The agent's response to the prompt.
            check_response(str): The response to check against.

        Returns:
            str
        """

        assert self.endpoint is not None, "Endpoint is not set."

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            prompt=(prompts.AGREEMENT_SYSTEM_PROMPT %
                    (prompt, check_response)) + response
        )

    def comprehensiveness_with_cot_reasons(self, source: str,
                                           summary: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that tries to distill main points
        and compares a summary against those main points. This feedback function
        only has a chain of thought implementation as it is extremely important
        in function assessment.

        **Usage:**
        ```python
        feedback = Feedback(provider.comprehensiveness_with_cot_reasons).on_input_output()
        ```

        Args:
            source (str): Text corresponding to source material. 
            summary (str): Text corresponding to a summary.

        Returns:
            float: A value between 0.0 (main points missed) and 1.0 (no main
            points missed).
        """

        system_prompt = str.format(
            prompts.COMPREHENSIVENESS_PROMPT, source=source, summary=summary
        )
        return self.generate_score_and_reasons(system_prompt)

    def summarization_with_cot_reasons(self, source: str,
                                       summary: str) -> Tuple[float, Dict]:
        """
        Summarization is deprecated in place of comprehensiveness. Defaulting to comprehensiveness_with_cot_reasons.
        """
        logger.warning(
            "summarization_with_cot_reasons is deprecated, please use comprehensiveness_with_cot_reasons instead."
        )
        return self.comprehensiveness_with_cot_reasons(source, summary)

    def stereotypes(self, prompt: str, response: str) -> float:
        """
        Uses chat completion model. A function that completes a template to
        check adding assumed stereotypes in the response when not present in the
        prompt.

        **Usage:**
        ```python
        feedback = Feedback(provider.stereotypes).on_input_output()
        ```

        Args:
            prompt (str): A text prompt to an agent. 
            response (str): The agent's response to the prompt.

        Returns:
            float: A value between 0.0 (no stereotypes assumed) and 1.0
            (stereotypes assumed).
        """

        system_prompt = str.format(
            prompts.STEREOTYPES_PROMPT, prompt=prompt, response=response
        )
        return self.generate_score(system_prompt)

    def stereotypes_with_cot_reasons(self, prompt: str,
                                     response: str) -> Tuple[float, Dict]:
        """
        Uses chat completion model. A function that completes a template to
        check adding assumed stereotypes in the response when not present in the
        prompt.

        **Usage:**
        ```python
        feedback = Feedback(provider.stereotypes).on_input_output()
        ```

        Args:
            prompt (str): A text prompt to an agent. 
            response (str): The agent's response to the prompt.

        Returns:
            float: A value between 0.0 (no stereotypes assumed) and 1.0
            (stereotypes assumed).
        """
        system_prompt = str.format(
            prompts.STEREOTYPES_PROMPT, prompt=prompt, response=response
        ) + prompts.COT_REASONS_TEMPLATE

        return self.generate_score_and_reasons(system_prompt)

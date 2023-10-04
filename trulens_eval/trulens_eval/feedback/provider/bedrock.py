import logging
import os

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import Provider
from trulens_eval.utils.generated import re_1_10_rating

import json

logger = logging.getLogger(__name__)


class Bedrock(Provider):
    model_id: str
    region_name: str

    def __init__(
        self, *args, model_id="amazon.titan-tg1-large", region_name="us-east-1", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        A set of AWS Feedback Functions.

        Parameters:

        - model_id (str, optional): The specific model id. Defaults to
          "amazon.titan-tg1-large".
        - region_name (str, optional): The specific AWS region name. Defaults to
          "us-east-1"

        - All other args/kwargs passed to the boto3 client constructor.
        """
        import boto3
        
        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict()
        self_kwargs.update(**kwargs)

        self_kwargs['model_id'] = model_id
        self_kwargs['region_name'] = region_name

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _create_chat_completion(self, prompt, *args, **kwargs):

        # NOTE(joshr): only tested with sso auth
        import boto3
        import json
        bedrock = boto3.client(service_name='bedrock-runtime')

        body = json.dumps({
            "inputText": prompt})

        modelId = self.model_id

        response = bedrock.invoke_model(body=body, modelId=modelId)

        response_body = json.loads(response.get('body').read()).get('results')[0]["outputText"]
        # text
        return response_body


    def _find_relevant_string(self, full_source, hypothesis):
        return self._create_chat_completion(
                prompt = 
                            str.format(
                                prompts.SYSTEM_FIND_SUPPORTING,
                                prompt=full_source,
                            ) + "\n" +
                            str.format(
                                prompts.USER_FIND_SUPPORTING,
                                response=hypothesis
                            )
            )

    def _summarized_groundedness(self, premise: str, hypothesis: str) -> float:
        """ A groundedness measure best used for summarized premise against simple hypothesis.
        This AWS Bedrock implementation uses information overlap prompts.

        Args:
            premise (str): Summarized source sentences.
            hypothesis (str): Single statement setnece.

        Returns:
            float: Information Overlap
        """
        return re_1_10_rating(
            self._create_chat_completion(
                    prompt=
                                str.format(
                                    prompts.LLM_GROUNDEDNESS,
                                    premise=premise,
                                    hypothesis=hypothesis,
                                )
                )
            ) / 10

    def _groundedness_doc_in_out(self, premise: str, hypothesis: str) -> str:
        """An LLM prompt using the entire document for premise and entire statement document for hypothesis

        Args:
            premise (str): A source document
            hypothesis (str): A statement to check

        Returns:
            str: An LLM response using a scorecard template
        """
        return self._create_chat_completion(
                prompt=
                            str.format(prompts.LLM_GROUNDEDNESS_FULL_SYSTEM,) + 
                            str.format(
                                prompts.LLM_GROUNDEDNESS_FULL_PROMPT,
                                premise=premise,
                                hypothesis=hypothesis
                            )
            )

    def qs_relevance(self, question: str, statement: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the relevance of the statement to the question.

        Parameters:
            question (str): A question being asked. statement (str): A statement
            to the question.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return re_1_10_rating(
            self._create_chat_completion(
                    prompt=
                                str.format(
                                    prompts.QS_RELEVANCE,
                                    question=question,
                                    statement=statement
                                )
                )
            ) / 10

    def relevance(self, prompt: str, response: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the relevance of the response to a prompt.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return re_1_10_rating(
            self._create_chat_completion(prompt = 
                                str.format(
                                    prompts.PR_RELEVANCE,
                                    prompt=prompt,
                                    response=response
                                )
                )
        ) / 10

    def sentiment(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the sentiment of some text.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """

        return re_1_10_rating(
            self._create_chat_completion(
                    system_prompt = prompts.SENTIMENT_SYSTEM_PROMPT,
                    human_prompt = text
                )
            )

    def model_agreement(self, prompt: str, response: str) -> float:
        """
        Uses AWS Titan Model. A function that gives AWS Titan the same
        prompt and gets a response, encouraging truthfulness. A second template
        is given to AWS Titan with a prompt that the original response is
        correct, and measures whether previous AWS Titan response is similar.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not in agreement" and 1
            being "in agreement".
        """
        logger.warning(
            "model_agreement has been deprecated. Use GroundTruthAgreement(ground_truth) instead."
        )
        aws_chat_response = self._create_chat_completion(
                system_prompt = prompts.CORRECT_SYSTEM_PROMPT,
                huamn_prompt = ""
            )
        agreement_txt = self._get_answer_agreement(
            prompt, response, aws_chat_response
        )
        return re_1_10_rating(agreement_txt) / 10

    def _langchain_evaluate(self, text: str, system_prompt: str) -> float:
        """
        Uses AWS Bedrock model. A general function that completes a
        template to evaluate different aspects of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent.
            system_prompt (str): The specific system prompt for evaluation.

        Returns:
            float: A value between 0 and 1, representing the evaluation.
        """

        return re_1_10_rating(
            self._create_chat_completion(
                system_prompt=system_prompt,
                human_prompt=text
            )
        ) / 10

    def conciseness(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the conciseness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not concise" and 1
            being "concise".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_CONCISENESS_PROMPT)
    
    def correctness(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the correctness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not correct" and 1
            being "correct".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_CORRECTNESS_PROMPT)

    def coherence(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the coherence of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not coherent" and 1
            being "coherent".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_COHERENCE_PROMPT)

    def harmfulness(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the harmfulness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "harmful" and 1
            being "not harmful".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_HARMFULNESS_PROMPT)

    def maliciousness(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the maliciousness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "malicious" and 1
            being "not malicious".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_MALICIOUSNESS_PROMPT)

    def helpfulness(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the helpfulness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not helpful" and 1
            being "helpful".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_HELPFULNESS_PROMPT)

    def controversiality(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the controversiality of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "controversial" and 1
            being "not controversial".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_CONTROVERSIALITY_PROMPT)

    def misogyny(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the misogyny of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "misogynist" and 1
            being "not misogynist".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_MISOGYNY_PROMPT)

    def criminality(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the criminality of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "criminal" and 1
            being "not criminal".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_CRIMINALITY_PROMPT)

    def insensitivity(self, text: str) -> float:
        """
        Uses AWS Bedrock model. A function that completes a
        template to check the insensitivity of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "insensitive" and 1
            being "not insensitive".
        """
        return self._langchain_evaluate(text, prompts.LANGCHAIN_INSENSITIVITY_PROMPT)

    def _get_answer_agreement(
        self, prompt, response, check_response
    ):
        bedrock_chat_response = self._create_chat_completion(
                system_prompt=
                            prompts.AGREEMENT_SYSTEM_PROMPT %
                            (prompt, response),
                human_prompt = check_response
            )
        return bedrock_chat_response

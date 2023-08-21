import logging

import openai
import os

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import Provider
from trulens_eval.feedback.provider.endpoint import OpenAIEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.keys import set_openai_key
from trulens_eval.utils.generated import re_1_10_rating

logger = logging.getLogger(__name__)


class OpenAI(Provider):
    model_engine: str
    endpoint: Endpoint

    def __init__(
        self, *args, endpoint=None, model_engine="gpt-3.5-turbo", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        A set of OpenAI Feedback Functions.

        Parameters:

        - model_engine (str, optional): The specific model version. Defaults to
          "gpt-3.5-turbo".

        - All other args/kwargs passed to OpenAIEndpoint constructor.
        """

        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs['model_engine'] = model_engine
        self_kwargs['endpoint'] = OpenAIEndpoint(*args, **kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

        set_openai_key()

    """
    def to_json(self) -> Dict:
        return Provider.to_json(self, model_engine=self.model_engine)
    """

    def _create_chat_completion(self, *args, **kwargs):
        return openai.ChatCompletion.create(*args, **kwargs)

    def _moderation(self, text: str):
        return self.endpoint.run_me(
            lambda: openai.Moderation.create(input=text)
        )

    def moderation_not_hate(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is hate
        speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "hate" and 1 being "not
            hate".
        """
        openai_response = self._moderation(text)
        return 1 - float(
            openai_response["results"][0]["category_scores"]["hate"]
        )

    def moderation_not_hatethreatening(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is
        threatening speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "threatening" and 1 being
            "not threatening".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["hate/threatening"]
        )

    def moderation_not_selfharm(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        self harm.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "self harm" and 1 being "not
            self harm".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["self-harm"]
        )

    def moderation_not_sexual(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is sexual
        speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "sexual" and 1 being "not
            sexual".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual"]
        )

    def moderation_not_sexualminors(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        sexual minors.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "sexual minors" and 1 being
            "not sexual minors".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual/minors"]
        )

    def moderation_not_violence(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        violence.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "violence" and 1 being "not
            violence".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence"]
        )

    def moderation_not_violencegraphic(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "graphic violence" and 1
            being "not graphic violence".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence/graphic"]
        )

    def _find_relevant_string(self, full_source, hypothesis):
        return self.endpoint.run_me(
            lambda: self._create_chat_completion(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            str.format(
                                prompts.SYSTEM_FIND_SUPPORTING,
                                prompt=full_source,
                            )
                    }, {
                        "role":
                            "user",
                        "content":
                            str.format(
                                prompts.USER_FIND_SUPPORTING,
                                response=hypothesis
                            )
                    }
                ]
            )["choices"][0]["message"]["content"]
        )

    def _summarized_groundedness(self, premise: str, hypothesis: str) -> float:
        """ A groundedness measure best used for summarized premise against simple hypothesis.
        This OpenAI implementation uses information overlap prompts.

        Args:
            premise (str): Summarized source sentences.
            hypothesis (str): Single statement setnece.

        Returns:
            float: Information Overlap
        """
        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    prompts.LLM_GROUNDEDNESS,
                                    premise=premise,
                                    hypothesis=hypothesis,
                                )
                        }
                    ]
                )["choices"][0]["message"]["content"]
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
        return self.endpoint.run_me(
            lambda: self._create_chat_completion(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            str.format(prompts.LLM_GROUNDEDNESS_FULL_SYSTEM,)
                    }, {
                        "role":
                            "user",
                        "content":
                            str.format(
                                prompts.LLM_GROUNDEDNESS_FULL_PROMPT,
                                premise=premise,
                                hypothesis=hypothesis
                            )
                    }
                ]
            )["choices"][0]["message"]["content"]
        )

    def qs_relevance(self, question: str, statement: str) -> float:
        """
        Uses OpenAI's Chat Completion App. A function that completes a
        template to check the relevance of the statement to the question.

        Parameters:
            question (str): A question being asked. statement (str): A statement
            to the question.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    prompts.QS_RELEVANCE,
                                    question=question,
                                    statement=statement
                                )
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def relevance(self, prompt: str, response: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the relevance of the response to a prompt.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    prompts.PR_RELEVANCE,
                                    prompt=prompt,
                                    response=response
                                )
                        },
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def sentiment(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the sentiment of some text.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.SENTIMENT_SYSTEM_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        )

    def model_agreement(self, prompt: str, response: str) -> float:
        """
        Uses OpenAI's Chat GPT Model. A function that gives Chat GPT the same
        prompt and gets a response, encouraging truthfulness. A second template
        is given to Chat GPT with a prompt that the original response is
        correct, and measures whether previous Chat GPT's response is similar.

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
        oai_chat_response = self.endpoint.run_me(
            lambda: self._create_chat_completion(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": prompts.CORRECT_SYSTEM_PROMPT
                    }, {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )["choices"][0]["message"]["content"]
        )
        agreement_txt = self._get_answer_agreement(
            prompt, response, oai_chat_response, self.model_engine
        )
        return re_1_10_rating(agreement_txt) / 10

    def conciseness(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the conciseness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not concise" and 1
            being "concise".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_CONCISENESS_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def correctness(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the correctness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not correct" and 1
            being "correct".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_CORRECTNESS_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def coherence(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the coherence of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not coherent" and 1
            being "coherent".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_COHERENCE_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def harmfulness(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the harmfulness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "harmful" and 1
            being "not harmful".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_HARMFULNESS_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def maliciousness(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the maliciousness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "malicious" and 1
            being "not malicious".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_MALICIOUSNESS_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def helpfulness(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the helpfulness of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not helpful" and 1
            being "helpful".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_HELPFULNESS_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def controversiality(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the controversiality of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "controversial" and 1
            being "not controversial".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_CONTROVERSIALITY_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def misogyny(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the misogyny of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "misogynist" and 1
            being "not misogynist".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_MISOGYNY_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def criminality(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the criminality of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "criminal" and 1
            being "not criminal".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_CRIMINALITY_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def insensitivity(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the insensitivity of some text. Prompt credit to Langchain Eval.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "insensitive" and 1
            being "not insensitive".
        """

        return re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts.LANGCHAIN_INSENSITIVITY_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def _get_answer_agreement(
        self, prompt, response, check_response, model_engine="gpt-3.5-turbo"
    ):
        oai_chat_response = self.endpoint.run_me(
            lambda: self._create_chat_completion(
                model=model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            prompts.AGREEMENT_SYSTEM_PROMPT %
                            (prompt, response)
                    }, {
                        "role": "user",
                        "content": check_response
                    }
                ]
            )["choices"][0]["message"]["content"]
        )
        return oai_chat_response


class AzureOpenAI(OpenAI):
    deployment_id: str

    def __init__(self, endpoint=None, **kwargs):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Wrapper to use Azure OpenAI. Please export the following env variables

        - OPENAI_API_BASE
        - OPENAI_API_VERSION
        - OPENAI_API_KEY

        Parameters:

        - model_engine (str, optional): The specific model version. Defaults to
          "gpt-35-turbo".
        - deployment_id (str): The specified deployment id
        """

        super().__init__(
            **kwargs
        )  # need to include pydantic.BaseModel.__init__

        set_openai_key()
        openai.api_type = "azure"
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")

    def _create_chat_completion(self, *args, **kwargs):
        """
        We need to pass `engine`
        """
        return super()._create_chat_completion(
            *args, deployment_id=self.deployment_id, **kwargs
        )

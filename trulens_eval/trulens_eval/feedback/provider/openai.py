import logging
from typing import Dict, Optional, Sequence

import openai as oai

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import OpenAIClient
from trulens_eval.feedback.provider.endpoint import OpenAIEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint

logger = logging.getLogger(__name__)


class OpenAI(LLMProvider):
    """
    Out of the box feedback functions calling OpenAI APIs.
    """

    # model_engine: str # LLMProvider

    endpoint: Endpoint

    def __init__(
        self, *args, endpoint=None, model_engine="gpt-3.5-turbo", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Create an OpenAI Provider with out of the box feedback functions.

        **Usage:**
        ```python
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()
        ```

        Args:
            model_engine (str): The OpenAI completion model. Defaults to
                `gpt-3.5-turbo`
            endpoint (Endpoint): Internal Usage for DB serialization
        """
        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs['model_engine'] = model_engine
        self_kwargs['endpoint'] = OpenAIEndpoint(*args, **kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    # LLMProvider requirement
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        if 'model' not in kwargs:
            kwargs['model'] = self.model_engine

        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.0

        if 'seed' not in kwargs:
            kwargs['seed'] = 123

        if prompt is not None:
            completion = self.endpoint.client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": prompt
                }], **kwargs
            )
        elif messages is not None:
            completion = self.endpoint.client.chat.completions.create(
                messages=messages, **kwargs
            )

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        return completion.choices[0].message.content

    def _moderation(self, text: str):
        # See https://platform.openai.com/docs/guides/moderation/overview .
        moderation_response = self.endpoint.run_me(
            lambda: self.endpoint.client.moderations.create(input=text)
        )
        return moderation_response.results[0]

    # TODEP
    def moderation_hate(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is hate
        speech.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_hate, higher_is_better=False
        ).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not hate) and 1.0 (hate).
        """
        openai_response = self._moderation(text)
        return float(openai_response.category_scores.hate)

    # TODEP
    def moderation_hatethreatening(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is
        threatening speech.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_hatethreatening, higher_is_better=False
        ).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not threatening) and 1.0 (threatening).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.hate_threatening)

    # TODEP
    def moderation_selfharm(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        self harm.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_selfharm, higher_is_better=False
        ).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not self harm) and 1.0 (self harm).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.self_harm)

    # TODEP
    def moderation_sexual(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is sexual
        speech.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_sexual, higher_is_better=False
        ).on_output()
        ```
        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not sexual) and 1.0 (sexual).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.sexual)

    # TODEP
    def moderation_sexualminors(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        sexual minors.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_sexualminors, higher_is_better=False
        ).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not sexual minors) and 1.0 (sexual
            minors).
        """

        openai_response = self._moderation(text)

        return float(oopenai_response.category_scores.sexual_minors)

    # TODEP
    def moderation_violence(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        violence.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_violence, higher_is_better=False
        ).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not violence) and 1.0 (violence).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.violence)

    # TODEP
    def moderation_violencegraphic(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_violencegraphic, higher_is_better=False
        ).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not graphic violence) and 1.0 (graphic
            violence).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.violence_graphic)

    # TODEP
    def moderation_harassment(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_harassment, higher_is_better=False
        ).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not harrassment) and 1.0 (harrassment).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.harassment)

    def moderation_harassment_threatening(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()

        feedback = Feedback(
            openai_provider.moderation_harassment_threatening, higher_is_better=False
        ).on_output()
        ```

        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not harrassment/threatening) and 1.0 (harrassment/threatening).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.harassment)


class AzureOpenAI(OpenAI):
    """Out of the box feedback functions calling AzureOpenAI APIs.
    Has the same functionality as OpenAI out of the box feedback functions.
    """

    def __init__(self, deployment_name: str, endpoint=None, **kwargs):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Wrapper to use Azure OpenAI. Please export the following env variables

        - AZURE_OPENAI_ENDPOINT
        - AZURE_OPENAI_API_KEY
        - OPENAI_API_VERSION

        **Usage:**
        ```python
        from trulens_eval.feedback.provider.openai import AzureOpenAI
        openai_provider = AzureOpenAI(deployment_name="...")
        ```

        Args:
            deployment_name (str, required): The name of the deployment.
            endpoint (Endpoint): Internal Usage for DB serialization
        """

        kwargs["client"] = OpenAIClient(client=oai.AzureOpenAI(**kwargs))
        super().__init__(
            endpoint=endpoint, model_engine=deployment_name, **kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _create_chat_completion(self, *args, **kwargs):
        """
        We need to pass `engine`
        """
        return super()._create_chat_completion(*args, **kwargs)

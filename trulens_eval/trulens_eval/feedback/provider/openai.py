import logging
import os
from typing import Dict, Mapping, Optional, Sequence

import openai
from openai import OpenAI

client = OpenAI()

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import OpenAIEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.keys import set_openai_key
from trulens_eval.utils.generated import re_0_10_rating

logger = logging.getLogger(__name__)


class OpenAI(LLMProvider):
    """Out of the box feedback functions calling OpenAI APIs.
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

        set_openai_key()

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

        if prompt is not None:
            comp = client.chat.completions.create(messages=[{
                "role": "system",
                "content": prompt
            }], **kwargs)
        elif messages is not None:
            comp = client.chat.completions.create(messages=messages, **kwargs)

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        assert isinstance(comp, dict)

        return comp["choices"][0]["message"]["content"]

    def _moderation(self, text: str):
        # See https://platform.openai.com/docs/guides/moderation/overview .
        return self.endpoint.run_me(
            lambda: client.moderations.create(input=text)
        )

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
        return float(openai_response["results"][0]["category_scores"]["hate"])

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

        return float(
            openai_response["results"][0]["category_scores"]["hate/threatening"]
        )

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

        return float(
            openai_response["results"][0]["category_scores"]["self-harm"]
        )

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

        return float(openai_response["results"][0]["category_scores"]["sexual"])

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

        return float(
            openai_response["results"][0]["category_scores"]["sexual/minors"]
        )

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

        return float(
            openai_response["results"][0]["category_scores"]["violence"]
        )

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
            float: A value between 0.0 (graphic violence) and 1.0 (not graphic
            violence).
        """
        openai_response = self._moderation(text)

        return float(
            openai_response["results"][0]["category_scores"]["violence/graphic"]
        )


class AzureOpenAI(OpenAI):
    """Out of the box feedback functions calling AzureOpenAI APIs.
    Has the same functionality as OpenAI out of the box feedback functions.
    """
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

        **Usage:**
        ```
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = AzureOpenAI(deployment_id="...")

        ```

        Args:
            model_engine (str, optional): The specific model version. Defaults
                to "gpt-35-turbo".
            deployment_id (str): The specified deployment id
            endpoint (Endpoint): Internal Usage for DB serialization
        """

        super().__init__(
            **kwargs
        )  # need to include pydantic.BaseModel.__init__

        set_openai_key()
        raise Exception("The 'openai.api_type' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_type="azure")'")
        raise Exception("The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base=os.getenv("OPENAI_API_BASE"))'")
        raise Exception("The 'openai.api_version' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_version=os.getenv("OPENAI_API_VERSION"))'")

    def _create_chat_completion(self, *args, **kwargs):
        """
        We need to pass `engine`
        """
        return super()._create_chat_completion(
            *args, deployment_id=self.deployment_id, **kwargs
        )

import logging
from typing import ClassVar, Dict, Optional, Sequence

import pydantic
from trulens.core.feedback import Endpoint
from trulens.core.utils.constants import CLASS_INFO
from trulens.core.utils.pace import Pace
from trulens.feedback import LLMProvider
from trulens.providers.openai.endpoint import OpenAIClient
from trulens.providers.openai.endpoint import OpenAIEndpoint

import openai as oai

logger = logging.getLogger(__name__)


class OpenAI(LLMProvider):
    """
    Out of the box feedback functions calling OpenAI APIs. Additionally, all feedback functions listed in the base [LLMProvider class][trulens.feedback.LLMProvider] can be run with OpenAI.

    Create an OpenAI Provider with out of the box feedback functions.

    Example:

        ```python
        from trulens.providers.openai import OpenAI
        openai_provider = OpenAI()
        ```

    Args:
        model_engine: The OpenAI completion model. Defaults to
            `gpt-4o-mini`

        **kwargs: Additional arguments to pass to the
            [OpenAIEndpoint][trulens.providers.openai.endpoint.OpenAIEndpoint]
            which are then passed to
            [OpenAIClient][trulens.providers.openai.endpoint.OpenAIClient]
            and finally to the OpenAI client.
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "gpt-4o-mini"

    # Endpoint cannot presently be serialized but is constructed in __init__
    # below so it is ok.
    endpoint: Endpoint = pydantic.Field(exclude=True)

    def __init__(
        self,
        *args,
        endpoint=None,
        pace: Optional[Pace] = None,
        rpm: Optional[int] = None,
        model_engine: Optional[str] = None,
        **kwargs: dict,
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE

        # Separate set of args for our attributes because only a subset go into
        # endpoint below.
        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs["model_engine"] = model_engine

        self_kwargs["endpoint"] = OpenAIEndpoint(
            *args, pace=pace, rpm=rpm, **kwargs
        )

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    # LLMProvider requirement
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs,
    ) -> str:
        if "model" not in kwargs:
            kwargs["model"] = self.model_engine

        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.0

        if "seed" not in kwargs:
            kwargs["seed"] = 123

        if messages is not None:
            completion = self.endpoint.client.chat.completions.create(
                messages=messages, **kwargs
            )

        elif prompt is not None:
            completion = self.endpoint.client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}], **kwargs
            )

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        return completion.choices[0].message.content

    def _moderation(self, text: str):
        # See https://platform.openai.com/docs/guides/moderation/overview .
        moderation_response = self.endpoint.run_in_pace(
            func=self.endpoint.client.moderations.create, input=text
        )
        return moderation_response.results[0]

    # TODEP
    def moderation_hate(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is hate
        speech.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_hate, higher_is_better=False
            ).on_output()
            ```

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

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_hatethreatening, higher_is_better=False
            ).on_output()
            ```

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

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_selfharm, higher_is_better=False
            ).on_output()
            ```

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

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_sexual, higher_is_better=False
            ).on_output()
            ```

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

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_sexualminors, higher_is_better=False
            ).on_output()
            ```

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not sexual minors) and 1.0 (sexual minors).
        """

        openai_response = self._moderation(text)

        return float(openai_response.category_scores.sexual_minors)

    # TODEP
    def moderation_violence(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        violence.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_violence, higher_is_better=False
            ).on_output()
            ```

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

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_violencegraphic, higher_is_better=False
            ).on_output()
            ```

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not graphic violence) and 1.0 (graphic violence).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.violence_graphic)

    # TODEP
    def moderation_harassment(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_harassment, higher_is_better=False
            ).on_output()
            ```

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not harassment) and 1.0 (harassment).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.harassment)

    def moderation_harassment_threatening(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_harassment_threatening, higher_is_better=False
            ).on_output()
            ```

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0.0 (not harassment/threatening) and 1.0 (harassment/threatening).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.harassment)


class AzureOpenAI(OpenAI):
    """
    !!! warning
        _Azure OpenAI_ does not support the _OpenAI_ moderation endpoint.
    Out of the box feedback functions calling AzureOpenAI APIs. Has the same
    functionality as OpenAI out of the box feedback functions, excluding the
    moderation endpoint which is not supported by Azure. Please export the
    following env variables. These can be retrieved from https://oai.azure.com/
    .

    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_API_KEY
    - OPENAI_API_VERSION

    Deployment name below is also found on the oai azure page.

    Example:
        ```python
        from trulens.providers.openai import AzureOpenAI
        openai_provider = AzureOpenAI(deployment_name="...")

        openai_provider.relevance(
            prompt="Where is Germany?",
            response="Poland is in Europe."
        ) # low relevance
        ```



    Args:
        deployment_name: The name of the deployment.
    """

    # Sent to our openai client wrapper but need to keep here as well so that it
    # gets dumped when jsonifying.
    deployment_name: str = pydantic.Field(alias="model_engine")

    def __init__(
        self,
        deployment_name: str,
        endpoint: Optional[Endpoint] = None,
        **kwargs: dict,
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        # Make a dict of args to pass to AzureOpenAI client. Remove any we use
        # for our needs. Note that model name / deployment name is not set in
        # that client and instead is an argument to each chat request. We pass
        # that through the super class's `_create_chat_completion`.
        client_kwargs = dict(kwargs)
        if CLASS_INFO in client_kwargs:
            del client_kwargs[CLASS_INFO]

        if "model_engine" in client_kwargs:
            # delete from client args
            del client_kwargs["model_engine"]
        else:
            # but include in provider args
            kwargs["model_engine"] = deployment_name

        kwargs["client"] = OpenAIClient(client=oai.AzureOpenAI(**client_kwargs))

        super().__init__(
            endpoint=None, **kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _create_chat_completion(self, *args, **kwargs):
        """
        We need to pass `engine`
        """
        return super()._create_chat_completion(*args, **kwargs)

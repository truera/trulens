import logging
from typing import Dict, Optional, Sequence, Tuple, Union

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import BedrockEndpoint
from trulens_eval.utils.generated import re_0_10_rating

logger = logging.getLogger(__name__)

# if using optional packages, check they are installed with this:
# OptionalImports(messages=REQUIREMENT_BEDROCK).assert_installed(...)

class Bedrock(LLMProvider):
    # LLMProvider requirement which we do not use:
    model_engine: str = "Bedrock"

    model_id: str
    endpoint: BedrockEndpoint

    def __init__(
        self, *args, model_id: str = "amazon.titan-tg1-large", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        A set of AWS Feedback Functions.

        Parameters:

        - model_id (str, optional): The specific model id. Defaults to
          "amazon.titan-tg1-large".

        - All other args/kwargs passed to BedrockEndpoint and subsequently
          to boto3 client constructor.
        """

        # SingletonPerName: return singleton unless client provided
        if hasattr(self, "model_id") and "client" not in kwargs:
            return

        # Pass kwargs to Endpoint. Self has additional ones.
        self_kwargs = dict()
        self_kwargs.update(**kwargs)

        self_kwargs['model_id'] = model_id

        self_kwargs['endpoint'] = BedrockEndpoint(*args, **kwargs)

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
        assert self.endpoint is not None
        assert prompt is not None, "Bedrock can only operate on `prompt`, not `messages`."

        import json

        body = json.dumps(
            {
                "inputText": prompt,
                "textGenerationConfig":
                    {
                        "maxTokenCount": 4096,
                        "stopSequences": [],
                        "temperature": 0,
                        "topP": 1
                    }
            }
        )
        # TODO: make textGenerationConfig available for user

        modelId = self.model_id

        response = self.endpoint.client.invoke_model(body=body, modelId=modelId)

        response_body = json.loads(response.get('body').read()
                                  ).get('results')[0]["outputText"]
        return response_body

    # overwrite base to use prompt instead of messages
    def _extract_score_and_reasons_from_response(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        normalize: float = 10.0
    ) -> Union[float, Tuple[float, Dict]]:
        """
        Extractor for LLM prompts. If CoT is used; it will look for
        "Supporting Evidence" template. Otherwise, it will look for the typical
        0-10 scoring.

        Args:
            system_prompt (str): A pre-formated system prompt

        Returns:
            The score and reason metadata if available.
        """
        response = self.endpoint.run_me(
            lambda: self._create_chat_completion(
                prompt=
                (system_prompt + user_prompt if user_prompt else system_prompt)
            )
        )
        if "Supporting Evidence" in response:
            score = 0.0
            supporting_evidence = None
            criteria = None
            for line in response.split('\n'):
                if "Score" in line:
                    score = re_0_10_rating(line) / normalize
                if "Criteria" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        criteria = ":".join(parts[1:]).strip()
                if "Supporting Evidence" in line:
                    supporting_evidence = line[
                        line.index("Supporting Evidence:") +
                        len("Supporting Evidence:"):].strip()
            reasons = {
                'reason':
                    (
                        f"{'Criteria: ' + str(criteria)}\n"
                        f"{'Supporting Evidence: ' + str(supporting_evidence)}"
                    )
            }
            return score, reasons
        else:
            return re_0_10_rating(response) / normalize

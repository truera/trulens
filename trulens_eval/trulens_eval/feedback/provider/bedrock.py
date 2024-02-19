import logging
from typing import Dict, Optional, Sequence, Tuple, Union

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import BedrockEndpoint
from trulens_eval.utils.generated import re_0_10_rating
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_BEDROCK

logger = logging.getLogger(__name__)

with OptionalImports(messages=REQUIREMENT_BEDROCK):
    # Here only to make sure we throw our message if bedrock optional packages
    # are not installed.
    import boto3

OptionalImports(messages=REQUIREMENT_BEDROCK).assert_installed(boto3)


class Bedrock(LLMProvider):
    """
    A set of AWS Feedback Functions.

    Parameters:

    - model_id (str, optional): The specific model id. Defaults to
        "amazon.titan-text-express-v1".

    - All other args/kwargs passed to BedrockEndpoint and subsequently
        to boto3 client constructor.
    """
    # LLMProvider requirement which we do not use:
    model_engine: str = "Bedrock"

    model_id: str
    endpoint: BedrockEndpoint

    def __init__(
        self,
        *args,
        model_id: str = "amazon.titan-text-express-v1",
        **kwargs
        # self, *args, model_id: str = "amazon.titan-text-express-v1", **kwargs
    ):


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

        if self.model_id.startswith("amazon"):
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
        elif self.model_id.startswith("anthropic"):
            body = json.dumps(
                {
                    "prompt": f"\n\nHuman:{prompt}\n\nAssistant:",
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens_to_sample": 4096
                }
            )
        elif self.model_id.startswith("cohere"):
            body = json.dumps(
                {
                    "prompt": prompt,
                    "temperature": 0,
                    "p": 1,
                    "max_tokens": 4096
                }
            )
        elif self.model_id.startswith("ai21"):
            body = json.dumps(
                {
                    "prompt": prompt,
                    "temperature": 0,
                    "topP": 1,
                    "maxTokens": 8191
                }
            )
        else:
            raise NotImplementedError(f"The model selected, {self.model_id}, is not yet implemented as a feedback provider")

        # TODO: make textGenerationConfig available for user

        modelId = self.model_id

        accept = "application/json"
        content_type = "application/json"

        response = self.endpoint.client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=content_type)
        
        if self.model_id.startswith("amazon"):
            response_body = json.loads(response.get('body').read()
                                    ).get('results')[0]["outputText"]

        if self.model_id.startswith("anthropic"):
            response_body = json.loads(response.get('body').read()
                                    ).get('completion')

        if self.model_id.startswith("cohere"):
            response_body = json.loads(response.get('body').read()
                                    ).get('generations')[0]["text"]

        if self.model_id.startswith("ai21"):
            response_body = json.loads(response.get('body').read()
                                    ).get('completions')[0].get('data').get('text')

        return response_body

    # overwrite base to use prompt instead of messages
    def generate_score(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        normalize: float = 10.0
    ) -> float:
        """
        Extractor for LLM prompts. If CoT is used; it will look for
        "Supporting Evidence" template. Otherwise, it will look for the typical
        0-10 scoring.

        Args:
            system_prompt (str): A pre-formated system prompt

        Returns:
            The score and reason metadata if available.
        """

        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            prompt=(
                system_prompt + user_prompt if user_prompt else system_prompt
            )
        )

        return re_0_10_rating(response) / normalize

    # overwrite base to use prompt instead of messages
    def generate_score_and_reasons(
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
        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            prompt=(
                system_prompt + user_prompt if user_prompt else system_prompt
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

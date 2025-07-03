import json
import logging
from typing import ClassVar, Dict, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel
from trulens.feedback import generated as feedback_generated
from trulens.feedback import llm_provider
from trulens.providers.bedrock import endpoint as bedrock_endpoint

logger = logging.getLogger(__name__)


class Bedrock(llm_provider.LLMProvider):
    """A set of AWS Feedback Functions.

    Args:
        model_id: The specific model id. Defaults to
            "amazon.titan-text-express-v1".

        *args: args passed to BedrockEndpoint and subsequently to boto3 client
            constructor.

        **kwargs: kwargs passed to BedrockEndpoint and subsequently to boto3
            client constructor.
    """

    DEFAULT_MODEL_ID: ClassVar[str] = "amazon.titan-text-express-v1"

    # LLMProvider requirement which we do not use:
    model_engine: str = "Bedrock"

    model_id: str
    endpoint: bedrock_endpoint.BedrockEndpoint

    def __init__(
        self,
        *args,
        model_id: Optional[str] = None,
        **kwargs,
        # self, *args, model_id: str = "amazon.titan-text-express-v1", **kwargs
    ):
        if model_id is None:
            model_id = self.DEFAULT_MODEL_ID

        # Pass kwargs to Endpoint. Self has additional ones.
        self_kwargs = dict()
        self_kwargs.update(**kwargs)

        self_kwargs["model_id"] = model_id

        self_kwargs["endpoint"] = bedrock_endpoint.BedrockEndpoint(
            *args, **kwargs
        )

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    # LLMProvider requirement
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        assert self.endpoint is not None

        if messages:
            messages_str = " ".join([
                f"{message['role']}: {message['content']}"
                for message in messages
            ])
        elif prompt:
            messages_str = prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' must be supplied.")

        if self.model_id.startswith("amazon"):
            body = json.dumps({
                "inputText": messages_str,
                "textGenerationConfig": {
                    "maxTokenCount": 4095,
                    "stopSequences": [],
                    "temperature": 0,
                    "topP": 1,
                },
            })
        elif self.model_id.startswith("anthropic"):
            if not messages:
                raise ValueError(
                    "`messages` argument must be supplied for Anthropic Bedrock models."
                )
            if messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
            _messages = messages[1:] if len(messages) > 1 else []

            body = json.dumps({
                "system": system_prompt,
                "messages": _messages,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 4095,
                "anthropic_version": "bedrock-2023-05-31",
            })
        elif self.model_id.startswith("cohere"):
            body = json.dumps({
                "prompt": messages_str,
                "temperature": 0,
                "p": 1,
                "max_tokens": 4095,
            })
        elif self.model_id.startswith("ai21"):
            body = json.dumps({
                "prompt": messages_str,
                "temperature": 0,
                "topP": 1,
                "maxTokens": 8191,
            })

        elif self.model_id.startswith("mistral"):
            body = json.dumps({
                "prompt": messages_str,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 4095,
            })

        elif self.model_id.startswith("meta"):
            body = json.dumps({
                "prompt": messages_str,
                "temperature": 0,
                "top_p": 1,
                "max_gen_len": 2047,
            })
        else:
            raise NotImplementedError(
                f"The Bedrock model selected, `{self.model_id}`, is not yet implemented as a feedback provider"
            )

        # TODO: make textGenerationConfig available for user

        modelId = self.model_id

        accept = "application/json"
        content_type = "application/json"

        response = self.endpoint.client.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=content_type
        )

        if self.model_id.startswith("amazon"):
            response_body = json.loads(response.get("body").read()).get(
                "results"
            )[0]["outputText"]

        elif self.model_id.startswith("anthropic"):
            response_body = json.loads(response.get("body").read()).get(
                "content"
            )[0]["text"]

        elif self.model_id.startswith("cohere"):
            response_body = json.loads(response.get("body").read()).get(
                "generations"
            )[0]["text"]

        elif self.model_id.startswith("mistral"):
            response_body = json.loads(response.get("body").read()).get(
                "output"
            )[0]["text"]
        elif self.model_id.startswith("meta"):
            response_body = json.loads(response.get("body").read()).get(
                "generation"
            )
        elif self.model_id.startswith("ai21"):
            response_body = (
                json.loads(response.get("body").read())
                .get("completions")[0]
                .get("data")
                .get("text")
            )

        return response_body

    # overwrite base to use prompt instead of messages
    def generate_score(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> float:
        """
        Base method to generate a score only, used for evaluation.

        Args:
            system_prompt: A pre-formatted system prompt.
            user_prompt: An optional user prompt.
            min_score_val: The minimum score value. Default is 0.
            max_score_val: The maximum score value. Default is 3.
            temperature: The temperature value for LLM score generation. Default is 0.0.

        Returns:
            The score on a 0-1 scale.
        """

        if temperature != 0.0:
            logger.warning(
                "The `temperature` argument is ignored for Bedrock provider."
            )

        llm_messages = [{"role": "system", "content": system_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})

        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion, messages=llm_messages
        )

        return (
            feedback_generated.re_configured_rating(response) - min_score_val
        ) / (max_score_val - min_score_val)

    # overwrite base to use prompt instead of messages
    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Union[float, Tuple[float, Dict]]:
        """
        Base method to generate a score and reason, used for evaluation.

        Args:
            system_prompt: A pre-formatted system prompt.
            user_prompt: An optional user prompt.
            min_score_val: The minimum score value. Default is 0.
            max_score_val: The maximum score value. Default is 3.
            temperature: The temperature value for LLM score generation. Default is 0.0.

        Returns:
            The score on a 0-1 scale.

            Reason metadata if returned by the LLM.
        """

        if temperature != 0.0:
            logger.warning(
                "The `temperature` argument is ignored for Bedrock provider."
            )

        llm_messages = [{"role": "system", "content": system_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})

        response = self.endpoint.run_in_pace(
            func=self._create_chat_completion, messages=llm_messages
        )
        if "Supporting Evidence" in response:
            score = 0.0
            supporting_evidence = None
            criteria = None
            for line in response.split("\n"):
                if "Score" in line:
                    score = (
                        feedback_generated.re_configured_rating(
                            line,
                            min_score_val=min_score_val,
                            max_score_val=max_score_val,
                        )
                        - min_score_val
                    ) / (max_score_val - min_score_val)
                if "Criteria" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        criteria = ":".join(parts[1:]).strip()
                if "Supporting Evidence" in line:
                    supporting_evidence = line[
                        line.index("Supporting Evidence:")
                        + len("Supporting Evidence:") :
                    ].strip()
            reasons = {
                "reason": (
                    f"{'Criteria: ' + str(criteria)}\n"
                    f"{'Supporting Evidence: ' + str(supporting_evidence)}"
                )
            }
            return score, reasons
        else:
            return (
                feedback_generated.re_configured_rating(
                    response,
                    min_score_val=min_score_val,
                    max_score_val=max_score_val,
                )
                - min_score_val
            ) / (max_score_val - min_score_val)

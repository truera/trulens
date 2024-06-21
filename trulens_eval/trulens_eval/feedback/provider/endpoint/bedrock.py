import inspect
import logging
import pprint
from typing import Any, Callable, ClassVar, Iterable, Optional

import pydantic

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.feedback.provider.endpoint.base import INSTRUMENT
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_BEDROCK
from trulens_eval.utils.python import safe_hasattr

with OptionalImports(messages=REQUIREMENT_BEDROCK) as opt:
    import boto3
    import botocore
    from botocore.client import ClientCreator

# check that the optional imports are not dummies:
opt.assert_installed([boto3, botocore])

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class BedrockCallback(EndpointCallback):

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def handle_generation_chunk(self, response: Any) -> None:
        super().handle_generation_chunk(response)

        # Example chunk:
        """
        {'chunk': {
            'bytes': b'''{"outputText":"\\nHello! I am a computer program designed to assist you. How can I help you today?",
                 "index":0,
                 "totalOutputTextTokenCount":21,
                 "completionReason":"FINISH",
                 "inputTextTokenCount":3,
                 "amazon-bedrock-invocationMetrics":{
                     "inputTokenCount":3,
                     "outputTokenCount":21,
                     "invocationLatency":1574,
                     "firstByteLatency":1574
                }}'''}}
        """

        chunk = response.get("chunk")
        if chunk is None:
            return

        data = chunk.get("bytes")
        if data is None:
            return

        import json
        data = json.loads(data.decode())

        metrics = data.get("amazon-bedrock-invocationMetrics")
        # Hopefully metrics are given only once at the last chunk so the below
        # adds are correct.
        if metrics is None:
            return

        output_tokens = metrics.get('outputTokenCount')
        if output_tokens is not None:
            self.cost.n_completion_tokens += int(output_tokens)
            self.cost.n_tokens += int(output_tokens)

        input_tokens = metrics.get('inputTokenCount')
        if input_tokens is not None:
            self.cost.n_prompt_tokens += int(input_tokens)
            self.cost.n_tokens += int(input_tokens)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)

        # Example response for completion:
        """
{'ResponseMetadata': {'HTTPHeaders': {'connection': 'keep-alive',
                                      'content-length': '181',
                                      'content-type': 'application/json',
                                      'date': 'Mon, 04 Dec 2023 23:25:27 GMT',
                                      'x-amzn-bedrock-input-token-count': '3',
                                      'x-amzn-bedrock-invocation-latency': '984',
                                      'x-amzn-bedrock-output-token-count': '20',
                      'HTTPStatusCode': 200,
                      'RetryAttempts': 0},
 'body': <botocore.response.StreamingBody object at 0x2bb6ae250>,
 'contentType': 'application/json'}
 """

        # NOTE(piotrm) LangChain does not currently support cost tracking for
        # Bedrock. We can at least count successes and tokens visible in the
        # example output above.

        was_success = False

        if response is not None:
            metadata = response.get("ResponseMetadata")
            if metadata is not None:
                status = metadata.get("HTTPStatusCode")
                if status is not None and status == 200:
                    was_success = True

                    headers = metadata.get("HTTPHeaders")
                    if headers is not None:
                        output_tokens = headers.get(
                            'x-amzn-bedrock-output-token-count'
                        )
                        if output_tokens is not None:
                            self.cost.n_completion_tokens += int(output_tokens)
                            self.cost.n_tokens += int(output_tokens)

                        input_tokens = headers.get(
                            'x-amzn-bedrock-input-token-count'
                        )
                        if input_tokens is not None:
                            self.cost.n_prompt_tokens += int(input_tokens)
                            self.cost.n_tokens += int(input_tokens)

        if was_success:
            self.cost.n_successful_requests += 1

        else:
            logger.warning(
                f"Could not parse bedrock response outcome to track usage.\n"
                f"{pp.pformat(response)}"
            )


class BedrockEndpoint(Endpoint):
    """
    Bedrock endpoint.
    
    Instruments `invoke_model` and `invoke_model_with_response_stream` methods
    created by `boto3.ClientCreator._create_api_method`.

    Args:
        region_name (str, optional): The specific AWS region name.
            Defaults to "us-east-1"

    """

    region_name: str

    # class not statically known
    client: Any = pydantic.Field(None, exclude=True)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, name="bedrock", **kwargs)

    def __str__(self) -> str:
        return f"BedrockEndpoint(region_name={self.region_name})"

    def __repr__(self) -> str:
        return f"BedrockEndpoint(region_name={self.region_name})"

    def __init__(
        self,
        *args,
        name: str = "bedrock",
        region_name: str = "us-east-1",
        **kwargs
    ):

        # SingletonPerName behaviour but only if client not provided.
        if hasattr(self, "region_name") and "client" not in kwargs:
            return

        # For constructing BedrockClient below:
        client_kwargs = {k: v for k, v in kwargs.items()}  # copy
        client_kwargs['region_name'] = region_name

        kwargs['region_name'] = region_name

        # for Endpoint, SingletonPerName:
        kwargs['name'] = name
        kwargs['callback_class'] = BedrockCallback

        super().__init__(*args, **kwargs)

        # Note here was are instrumenting a method that outputs a function which
        # we also want to instrument:
        if not safe_hasattr(ClientCreator._create_api_method, INSTRUMENT):
            self._instrument_class_wrapper(
                ClientCreator,
                wrapper_method_name="_create_api_method",
                wrapped_method_filter=lambda f: f.__name__ in
                ["invoke_model", "invoke_model_with_response_stream"]
            )

        if 'client' in kwargs:
            # `self.client` should be already set by super().__init__.

            if not safe_hasattr(self.client.invoke_model, INSTRUMENT):
                # If they user instantiated the client before creating our
                # endpoint, the above instrumentation will not have attached our
                # instruments. Do it here instead:
                self._instrument_class(type(self.client), "invoke_model")
                self._instrument_class(
                    type(self.client), "invoke_model_with_response_stream"
                )

        else:
            # This one will be instrumented by our hacks onto _create_api_method above:

            self.client = boto3.client(
                service_name='bedrock-runtime', **client_kwargs
            )

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:

        if func.__name__ == "invoke_model":
            self.global_callback.handle_generation(response=response)
            if callback is not None:
                callback.handle_generation(response=response)

        elif func.__name__ == "invoke_model_with_response_stream":
            self.global_callback.handle_generation(response=response)
            if callback is not None:
                callback.handle_generation(response=response)

            body = response.get("body")
            if body is not None and isinstance(body, Iterable):
                for chunk in body:
                    self.global_callback.handle_generation_chunk(response=chunk)
                    if callback is not None:
                        callback.handle_generation_chunk(response=chunk)

            else:
                logger.warning(
                    "No iterable body found in `invoke_model_with_response_stream` response."
                )

        else:

            logger.warning(f"Unhandled wrapped call to %s.", func.__name__)

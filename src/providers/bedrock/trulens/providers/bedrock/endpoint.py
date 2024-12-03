import inspect
import logging
import pprint
from typing import Any, Callable, ClassVar, Iterable, Optional, Sequence

import boto3
from botocore.client import ClientCreator
import pydantic
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.utils import python as python_utils
from trulens.experimental.otel_tracing import _feature as otel_tracing_feature

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class BedrockCallback(core_endpoint.EndpointCallback):
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

        if not isinstance(response, dict):
            return

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

        output_tokens = metrics.get("outputTokenCount")
        if output_tokens is not None:
            self.cost.n_completion_tokens += int(output_tokens)
            self.cost.n_tokens += int(output_tokens)

        input_tokens = metrics.get("inputTokenCount")
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
                            "x-amzn-bedrock-output-token-count"
                        )
                        if output_tokens is not None:
                            self.cost.n_completion_tokens += int(output_tokens)
                            self.cost.n_tokens += int(output_tokens)

                        input_tokens = headers.get(
                            "x-amzn-bedrock-input-token-count"
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


if otel_tracing_feature._FeatureSetup.are_optionals_installed():
    from trulens.experimental.otel_tracing.core.feedback import (
        endpoint as experimental_core_endpoint,
    )

    class _WrapperBedrockEndpointCallback(
        experimental_core_endpoint._WrapperEndpointCallback
    ):
        """EXPERIMENTAL(otel_tracing): Callbacks for instrumented methods in
        Bedrock's boto3 to account for costs."""

        model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

        def __init__(self, func: Callable, **kwargs):
            super().__init__(func=func, **kwargs)

            # Assume bedrock only has completion requests.
            self.cost.n_completion_requests += 1

        def on_endpoint_response(self, response: Any) -> None:
            func = self.func

            if func.__name__ == "invoke_model":
                self.on_endpoint_generation(response=response)

            elif func.__name__ == "invoke_model_with_response_stream":
                self.on_endpoint_generation(response=response)

                body = response.get("body")
                if body is not None:
                    if isinstance(body, Sequence):
                        # NOTE(piotrm): Intentionally checking for Sequence instead
                        # of Iterable to be sure that iterating over it will not
                        # steal it from the downstream users.

                        for chunk in body:
                            self.on_endpoint_generation_chunk(response=chunk)
                    else:
                        # TODO: Wrap iterable here.
                        logger.warning(
                            "Cannot safely iterate body in `invoke_model_with_response_stream` response."
                        )
                else:
                    logger.warning(
                        "No stream body found in `invoke_model_with_response_stream` response."
                    )

            else:
                logger.warning("Unhandled wrapped call to %s.", func.__name__)

        def on_endpoint_generation_chunk(self, response: Any) -> None:
            """Handle stream chunk.

            Example chunk:
            ```json
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
            ```
            """
            super().on_endpoint_generation_chunk(response)

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

            output_tokens = metrics.get("outputTokenCount")
            if output_tokens is not None:
                self.cost.n_completion_tokens += int(output_tokens)
                self.cost.n_tokens += int(output_tokens)

            input_tokens = metrics.get("inputTokenCount")
            if input_tokens is not None:
                self.cost.n_prompt_tokens += int(input_tokens)
                self.cost.n_tokens += int(input_tokens)

        def on_endpoint_generation(self, response: Any) -> None:
            """Handle completion generation.

            Example response for completion:
            ```json
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
            ```
            """

            super().on_endpoint_generation(response)

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
                                "x-amzn-bedrock-output-token-count"
                            )
                            if output_tokens is not None:
                                self.cost.n_completion_tokens += int(
                                    output_tokens
                                )
                                self.cost.n_tokens += int(output_tokens)

                            input_tokens = headers.get(
                                "x-amzn-bedrock-input-token-count"
                            )
                            if input_tokens is not None:
                                self.cost.n_prompt_tokens += int(input_tokens)
                                self.cost.n_tokens += int(input_tokens)

            if was_success:
                self.cost.n_successful_requests += 1

            else:
                logger.warning(
                    "Could not parse bedrock response outcome to track usage.\n%s",
                    pp.pformat(response),
                )


class BedrockEndpoint(core_endpoint.Endpoint):
    """Bedrock endpoint.

    Instruments `invoke_model` and `invoke_model_with_response_stream` methods
    created by `boto3.ClientCreator._create_api_method`.

    Args:
        region_name (str, optional): The specific AWS region name.
            Defaults to "us-east-1"

    """

    _experimental_wrapper_callback_class = (
        _WrapperBedrockEndpointCallback
        if otel_tracing_feature._FeatureSetup.are_optionals_installed()
        else None
    )

    region_name: str

    # class not statically known
    client: Any = pydantic.Field(None, exclude=True)

    def __str__(self) -> str:
        return f"BedrockEndpoint(region_name={self.region_name})"

    def __repr__(self) -> str:
        return f"BedrockEndpoint(region_name={self.region_name})"

    def __init__(
        self,
        *args,
        region_name: str = "us-east-1",
        **kwargs,
    ):
        # For constructing BedrockClient below:
        client_kwargs = {k: v for k, v in kwargs.items()}  # copy
        client_kwargs["region_name"] = region_name

        kwargs["region_name"] = region_name

        # for Endpoint
        kwargs["callback_class"] = BedrockCallback

        super().__init__(*args, **kwargs)

        # Note here was are instrumenting a method that outputs a function which
        # we also want to instrument:
        if not python_utils.safe_hasattr(
            ClientCreator._create_api_method, core_endpoint.INSTRUMENT
        ):
            self._instrument_class_wrapper(
                ClientCreator,
                wrapper_method_name="_create_api_method",
                wrapped_method_filter=lambda f: f.__name__
                in ["invoke_model", "invoke_model_with_response_stream"],
            )

        if "client" in kwargs:
            # `self.client` should be already set by super().__init__.

            if not python_utils.safe_hasattr(
                self.client.invoke_model, core_endpoint.INSTRUMENT
            ):
                # If they user instantiated the client before creating our
                # endpoint, the above instrumentation will not have attached our
                # instruments. Do it here instead:
                self._instrument_class(type(self.client), "invoke_model")
                self._instrument_class(
                    type(self.client), "invoke_model_with_response_stream"
                )

        else:
            # This one will be instrumented by our hacks onto _create_api_method
            # above:

            self.client = boto3.client(
                service_name="bedrock-runtime", **client_kwargs
            )

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> None:
        # TODELETE(otel_tracing). Delete once otel_tracing is no longer
        # experimental.

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
            logger.warning("Unhandled wrapped call to %s.", func.__name__)

        return response

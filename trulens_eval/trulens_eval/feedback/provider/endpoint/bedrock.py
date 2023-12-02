import inspect
import logging
import pprint
from typing import Any, Callable, Dict, List, Optional

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.utils.pyschema import WithClassInfo


logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


class BedrockCallback(EndpointCallback):

    class Config:
        arbitrary_types_allowed = True

    def handle_classification(self, response: Dict) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)


class BedrockEndpoint(Endpoint, WithClassInfo):
    """
    Bedrock endpoint. Instruments "completion" methods in bedrock.* classes.
    """

    region_name: str

    def __new__(cls, *args, **kwargs):
        print("BedrockEndpoint __new__")
        return super().__new__(cls, *args, name="bedrock", **kwargs)

    def __init__(self, *args, region_name: str,  **kwargs):
        # SingletonPerName
        if hasattr(self, "region_name"):
            return

        kwargs['region_name'] = region_name

        # for Endpoint, SingletonPerName:
        kwargs['name'] = "bedrock"
        kwargs['callback_class'] = BedrockCallback

        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)

        import boto3
        # The BedrockRuntime class does not exist statically. I believe it is
        # created from api descriptions. Here we make sure it is created before
        # we instrument it.
        # cls = boto3.client("bedrock-runtime").__class__
        # self._instrument_class(cls, "invoke_model")
        
        from botocore.client import ClientCreator
        # Note here was are instrumenting a method that outputs a function which
        # we also want to instrument:
        self._instrument_class_wrapper(
            ClientCreator,
            wrapper_method_name="_create_api_method",
            wrapped_method_name="invoke_model"
        )
        


    def handle_wrapped_call(
        self, 
        func: Callable, 
        bindings: inspect.BoundArguments, 
        response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:

        model_name = ""
        if 'model' in bindings.kwargs:
            model_name = bindings.kwargs['model']

        results = None
        if "results" in response:
            results = response['results']

        counted_something = False

        if 'usage' in response:
            counted_something = True
            usage = response['usage']

            self.global_callback.handle_generation(response=usage)

            if callback is not None:
                callback.handle_generation(response=usage)

        if not counted_something:
            logger.warning(
                f"Unrecognized bedrock response format. It did not have usage information:\n"
                + pp.pformat(response)
            )

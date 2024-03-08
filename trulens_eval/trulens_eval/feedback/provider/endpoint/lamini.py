import inspect
import logging
import pprint
from typing import Any, Callable, ClassVar, Optional

import pydantic

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LAMINI

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()

with OptionalImports(messages=REQUIREMENT_LAMINI):
    # Here only so we can throw the proper error if lamini is not installed.
    import lamini

OptionalImports(messages=REQUIREMENT_LAMINI).assert_installed(lamini)

class LaminiCallback(EndpointCallback):
    """Handlers for Lamini responses to track costs/performance/etc.
    
    !!! NOTE:
        Lamini does not currently produce any usage information in its responses
        so this class is not yet useful.
    """

    model_config: ClassVar[dict] = {'arbitrary_types_allowed': True}

    def handle_classification(self, response: pydantic.BaseModel) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: pydantic.BaseModel) -> None:
        """Get the usage information from lamini response."""
        super().handle_generation(response)

class LaminiEndpoint(Endpoint):
    """Lamini endpoint."""

    def __init__(self, **kwargs):
        if hasattr(self, "name"):
            # singleton already made
            return

        kwargs['name'] = "lamini"
        kwargs['callback_class'] = LaminiCallback

        super().__init__(**kwargs)

        self._instrument_module_members(lamini, "generate")

    def __new__(cls, **kwargs):
        # Problem here if someone uses lamini with different providers. Only a
        # single one will be made. Cannot make a fix just here as
        # track_all_costs creates endpoints via the singleton mechanism.

        return super(Endpoint, cls).__new__(cls, name="lamini")

    def handle_wrapped_call(
        self, func: Callable, bindings: inspect.BoundArguments, response: Any,
        callback: Optional[EndpointCallback]
    ) -> None:

        # Lamini does not currently produce any usage information as part of a response.

        self.global_callback.handle_generation(response=response)

        if callback is not None:
            callback.handle_generation(response=response)


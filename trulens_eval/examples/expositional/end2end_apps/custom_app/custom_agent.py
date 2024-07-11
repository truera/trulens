from typing import Any, Optional

from examples.expositional.end2end_apps.custom_app import dummy

from trulens_eval.tru_custom_app import instrument
from trulens_eval.tru_custom_app import TruCustomApp


class CustomAgent(dummy.Dummy):
    """Dummy agent implementation.
    
    This agent serves to demonstrate both the span type for agents but also the
    use of _TruLens-Eval_ internally in an app that is instrumented using
    _TruLens-Eval_.
    """

    DEFAULT_USE_APP: bool = False
    """Whether to use a custom app internally.
    
    This will result in a complex stack where a custom app calls an agent which
    itself calls an app and further agents. The agents inside the inner app do
    not include further apps to prevent an infinite structure.
    """

    DEFAULT_USE_RECORDER: bool = False
    """Whether to use a recorder internally to capture execution of itself."""

    def __init__(
        self,
        *args,
        app: Any, description: Optional[str] = None,
        use_app: Optional[bool] = None,
        use_recorder: Optional[bool] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if use_app is None:
            use_app = CustomAgent.DEFAULT_USE_APP
        if use_recorder is None:
            use_recorder = CustomAgent.DEFAULT_USE_RECORDER

        self.use_app = use_app
        self.use_recorder = use_rec
        self.description = description or "Custom Agent"
        self.app = app

        # TODO: This agent is meant to use a tru recorded internally but doing
        # so is presently broken in tracing. Renable this and the recording in
        # invoke once fixed.

        if self.use_app:
            self.tru_app = TruCustomApp(self.app, app_id=description)

        self.dummy_allocate()

    @instrument
    def invoke(self, data: str) -> str:
        """Invoke the dummy tool."""

        self.dummy_wait()

        # TODO: see prior note.

        if self.use_app:
            if self.use_recorder:
                with self.tru_app as recorder:
                    self.app.respond_to_query(query=data)
                return recorder.get().model_dump_json()
            else:
                return self.app.respond_to_query(query=data)
        else:
            return "Record placeholder"

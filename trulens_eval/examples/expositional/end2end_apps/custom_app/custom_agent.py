from typing import Any, Optional

from examples.expositional.end2end_apps.custom_app import dummy

from trulens_eval.schema import record as mod_record_schema
from trulens_eval.trace import span as mod_span
from trulens_eval.tru_custom_app import instrument
from trulens_eval.tru_custom_app import TruCustomApp


class CustomAgent(dummy.Dummy):
    """Dummy agent implementation.
    
    This agent serves to demonstrate both the span type for agents but also the
    use of _TruLens-Eval_ internally in an app that is instrumented using
    _TruLens-Eval_.
    """

    def __init__(
        self,
        *args,
        app: Any,
        description: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.description = description or "Custom Agent"
        self.app = app
        self.tru_app = TruCustomApp(self.app, app_id=description)

        self.dummy_allocate()

    @instrument
    def invoke(self, data: str) -> str:
        """Invoke the dummy tool."""

        self.dummy_wait()

        with self.tru_app as recorder:
            self.app.respond_to_query(query=data)

        return "Record placeholder"

        # return recorder.get().model_dump_json()

    @invoke.is_span(span_type=mod_span.SpanAgent)
    def set_agent_span(
        self,
        call: mod_record_schema.RecordAppCall,
        span: mod_span.SpanAgent
    ):
        span.description = self.description


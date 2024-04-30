import random
from typing import Callable, Optional

from examples.expositional.end2end_apps.custom_app.dummy import Dummy

from trulens_eval.schema import record as mod_record_schema
from trulens_eval.trace import span as mod_span
from trulens_eval.tru_custom_app import instrument

# Collect a few string methods to use as "tools".
str_maps = []


for name in dir(str):
    func = getattr(str, name)

    if isinstance(func, Callable):
        try:
            ret = func("test me ! LOLS")
            if isinstance(ret, str):
                str_maps.append(func)

        except Exception:
            pass


class CustomTool(Dummy):
    """Dummy tool implementation."""

    def __init__(
        self,
        *args,
        description: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if description is None:
            imp = random.choice(str_maps)
            description = imp.__doc__

        self.description = description
        self.imp = imp

        self.dummy_allocate()

    @instrument
    def invoke(self, data: str):
        """Invoke the dummy tool."""

        self.dummy_wait()

        return self.imp(data)
    
    @invoke.is_span(span_type=mod_span.SpanTool)
    def set_memory_span(
        self,
        call: mod_record_schema.RecordAppCall,
        span: mod_span.SpanMemory
    ):
        span.description = self.description

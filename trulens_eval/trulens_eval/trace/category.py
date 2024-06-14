"""Span categorization and category detection."""

from __future__ import annotations

import inspect
import logging
from typing import List, Optional, Sequence, Set, TypeVar


import opentelemetry.trace.span as ot_span

from trulens_eval import instruments as mod_instruments
from trulens_eval import trace as mod_trace
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.trace import span as mod_span
from trulens_eval.trace import tracer as mod_tracer
from trulens_eval.utils import containers as mod_containers_utils
from trulens_eval.utils import pyschema

T = TypeVar("T")

logger = logging.getLogger(__name__)

class Categorizer():
    """Categorizes RecordAppCalls into Spans of various types."""

    known_modules = set([
        "langchain",
        "llama_index",
        "nemoguardrails",
        "trulens_eval"
    ])

    @staticmethod
    def class_is(pycls: pyschema.Class) -> bool:
        """
        Determine whether the given class representation `pycls` is of the type to
        be viewed as this component type.
        """

        return True

    @staticmethod
    def innermost_base(
        bases: Optional[Sequence[pyschema.Class]] = None,
        among_modules: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Given a sequence of classes, return the first one which comes from one
        of the `among_modules`. You can use this to determine where ultimately
        the encoded class comes from in terms of langchain, llama_index, or
        trulens_eval even in cases they extend each other's classes. Returns
        None if no module from `among_modules` is named in `bases`.
        """
        if among_modules is None:
            among_modules = Categorizer.known_modules

        if bases is None:
            return None

        for base in bases:
            if "." in base.module.module_name:
                root_module = base.module.module_name.split(".")[0]
            else:
                root_module = base.module.module_name

            if root_module in among_modules:
                return root_module

        return None

    @staticmethod
    def span_of_call(
        call: mod_record_schema.RecordAppCall,
        tracer: mod_tracer.Tracer,
        context: Optional[ot_span.SpanContext] = None
    ) -> mod_span.Span:
        """Categorizes a [RecordAppCall][trulens_eval.schema.record.RecordAppCall] into a span.

        Args:
            call: The call to categorize.

            tracer: The tracer to create the span in.

            context: The context of the parent span if any.

        Returns:

            The span.
        """

        method = call.method()
        package_name = method.obj.cls.module.package_name

        subcategorizer = None
        subs = [
            LangChainCategorizer,
            LlamaIndexCategory,
            NemoGuardRailsCategorizer,
            TruLensCategorizer,
            CustomCategorizer
        ]

        if package_name is None:
            logger.warning("Unknown package.")

        for sub in subs:
            if sub.class_is(method.obj.cls):
                subcategorizer = sub
                break

        if subcategorizer is not None:
            return subcategorizer.span_of_call(
                call=call, tracer=tracer, context=context
            )

        span = tracer.new_span(
            name = method.name,
            cls = mod_span.SpanOther, # if no category known
            context = context
        )

        return span

    @staticmethod
    def spans_of_record(record: mod_record_schema.Record) -> List[mod_span.Span]:
        """Convert this record into a tracer with all of the calls populated as spans."""

        # Init with trace_id that is determined by record_id.
        tracer = mod_trace.tracer.Tracer(
            trace_id=mod_tracer.trace_id_of_string_id(record.record_id)
        )

        root_span = tracer.new_span(
            name="root",
            cls=mod_span.SpanRoot,
            start_time=mod_containers_utils.ns_timestamp_of_datetime(record.perf.start_time)
        )
        root_span.end(mod_containers_utils.ns_timestamp_of_datetime(record.perf.end_time))

        # TransSpanRecord fields
        root_span.record = record
        root_span.record_id = record.record_id

        method_to_span_map = {}

        for call in record.calls:
            method = call.top()
            span = Categorizer.span_of_call(
                call=call,
                tracer=tracer,
                context=root_span.context # might be changed below
            )

            method_to_span_map[method] = span

            # OTSpan fields
            span.start_timestamp = mod_containers_utils.ns_timestamp_of_datetime(call.perf.start_time)
            span.end(mod_containers_utils.ns_timestamp_of_datetime(call.perf.end_time))

            # TransSpanRecord fields
            span.record_id = record.record_id
            span.record = record

            # TransSpanRecordAppCall fields
            span.call = call
            span.inputs = call.args
            span.output = call.rets
            span.error = call.error

            # Add to traces. Not needed now but might be later.
            tracer.spans[span.context] = span

        # Update parent context links.
        for span in tracer.spans.values():
            if isinstance(span, mod_span.TransSpanRecordAppCall):

                parent_method = span.call.caller()
                if parent_method in method_to_span_map:

                    parent_span = method_to_span_map[parent_method]
                    span.parent_context = parent_span.context

        return list(tracer.spans.values())


class LangChainCategorizer(Categorizer):
    """Categorizer for _LangChain_ classes."""

    @staticmethod
    def class_is(pycls: pyschema.Class) -> bool:
        if Categorizer.innermost_base(pycls.bases) == "langchain":
            return True

        return False

    @staticmethod
    def span_of_call(
        call: mod_record_schema.RecordAppCall,
        tracer: mod_tracer.Tracer,
        context: Optional[ot_span.SpanContext] = None
    ) -> mod_span.Span:
        """Converts a call by a _LangChain_ class into the appropriate span."""
        
        pycls = call.method().obj.cls

        if pycls.noserio_issubclass(
            module_name="langchain.prompts.base",
            class_name="BasePromptTemplate"
        ) or pycls.noserio_issubclass(
            module_name="langchain.schema.prompt_template",
            class_name="BasePromptTemplate"
        ):  # langchain >= 0.230
            # Prompt
            pass

        elif pycls.noserio_issubclass(
            module_name="langchain.llms.base", class_name="BaseLLM"
        ):
            # LLM
            span = tracer.new_span(
                name = call.method().name,
                cls = mod_span.SpanLLM,
                context = context
            )
            span.model_name = "TBD"

        return tracer.new_span(
            name = call.method().name,
            cls = mod_span.SpanOther,
            context = context
        )

class LlamaIndexCategory(Categorizer):
    """Categorizer for _LlamaIndex_ classes."""

    @staticmethod
    def class_is(pycls: pyschema.Class) -> bool:
        if Categorizer.innermost_base(pycls.bases) == "llama_index":
            return True

        return False

    @staticmethod
    def span_of_call(
        call: mod_record_schema.RecordAppCall,
        tracer: mod_tracer.Tracer,
        context: Optional[ot_span.SpanContext] = None
    ) -> mod_span.Span:
        """Converts a call by a _LlamaIndex_ class into the appropriate span."""
        
        pycls = call.method().obj.cls

        if pycls.noserio_issubclass(
            module_name="llama_index.prompts.base", class_name="Prompt"
        ):
            # Prompt
            pass

        elif pycls.noserio_issubclass(
            module_name="llama_index.agent.types", class_name="BaseAgent"
        ):
            # Agent
            pass

        elif pycls.noserio_issubclass(
            module_name="llama_index.tools.types", class_name="BaseTool"
        ):
            # Tool
            pass

        elif pycls.noserio_issubclass(
            module_name="llama_index.llms.base", class_name="LLM"
        ):
            # LLM
            span = tracer.new_span(
                name = call.method().name,
                cls = mod_span.SpanLLM,
                context = context
            )
            span.model_name = "TBD"

        return tracer.new_span(
            name = call.method().name,
            cls = mod_span.SpanOther,
            context = context
        )


class NemoGuardRailsCategorizer(Categorizer):
    """Categorizer for _Nemo GuardRails_ classes."""

    @staticmethod
    def class_is(pycls: pyschema.Class) -> bool:
        if Categorizer.innermost_base(pycls.bases) == "nemoguardrails":
            return True

        return False

class TruLensCategorizer(Categorizer):
    """Categorizer for _TruLens-Eval_ classes."""

    @staticmethod
    def class_is(pycls: pyschema.Class) -> bool:
        if Categorizer.innermost_base(pycls.bases) == "trulens_eval":
            return True

        return False

class CustomCategorizer(Categorizer):
    """Categorizer for custom classes annotated with
    [instrument][trulens_eval.tru_custom_app.instrument] or related."""

    @staticmethod
    def class_is(pycls: pyschema.Class) -> bool:
        i = mod_instruments.Instrument()
        return i.to_instrument_class(pycls)
    
    @staticmethod
    def span_of_call(
        call: mod_record_schema.RecordAppCall,
        tracer: mod_tracer.Tracer,
        context: Optional[ot_span.SpanContext] = None
    ) -> mod_span.Span:
        """Converts a custom instrumentation-annotated method call into the appropriate span."""

        pymethod = call.method()
        pyfunc = pymethod.as_function()
        method_name = pymethod.name

        span_info = None

        if pyfunc in mod_instruments.Instrument.Default.SPANINFOS:
            for instance_or_class, si in mod_instruments.Instrument.Default.SPANINFOS[pyfunc].items():
                if isinstance(instance_or_class, type) and isinstance(si.spanner, staticmethod): # is_span decorator was on a staticmethod
                    span_info = (si.span_type, id(instance_or_class), instance_or_class, si.spanner)
                    break

                if id(instance_or_class) == pymethod.obj.id: # is_span decorated a method in a class later instantiated with object having this id
                    span_info = (si.span_type, id(instance_or_class), instance_or_class, si.spanner)
                    break

        if span_info is not None:
            cls = span_info[0].to_class()
        else:
            logger.warning("No custom spanner for: %s", pymethod)
            cls = mod_span.SpanOther

        span = tracer.new_span(
            name = method_name,
            context = context,
            cls=cls
        )
        if span_info is not None:
            _, obj_id, instance, spanner = span_info

            if obj_id is None:
                # staticmethod/function
                spanner(call=call, span=span)
            else:
                # method, requiring self
                spanner(self=instance, call=call, span=span)

        return span

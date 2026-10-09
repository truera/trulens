import asyncio
from collections import defaultdict
import functools
import gc
from typing import Callable
import unittest

from opentelemetry import baggage
from opentelemetry import trace
from opentelemetry.baggage import remove_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind
import pandas as pd
from trulens.core.otel.instrument import get_func_name
from trulens.core.otel.instrument import instrument
from trulens.core.otel.instrument import instrument_method
from trulens.core.otel.instrument import span_group
from trulens.core.otel.recording import Recording
from trulens.experimental.otel_tracing.core.session import (
    _set_up_tracer_provider,
)
from trulens.otel.semconv.trace import SpanAttributes


class TestOtelInstrument(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        instrument.enable_all_instrumentation()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        instrument.disable_all_instrumentation()
        return super().tearDownClass()

    def setUp(self) -> None:
        # Set up OTEL tracing.
        self.exporter = InMemorySpanExporter()
        _set_up_tracer_provider()
        self.span_processor = SimpleSpanProcessor(self.exporter)
        trace.get_tracer_provider().add_span_processor(self.span_processor)
        # We attach the following to the context so that any instrumented
        # functions will believe they are part of a recording but not the record
        # root.
        self.tokens = []
        self.tokens.append(
            context_api.attach(
                set_baggage("__trulens_recording__", Recording(None))
            )
        )
        self.tokens.append(
            context_api.attach(
                set_baggage(SpanAttributes.RECORD_ID, "test_record_id")
            )
        )
        return super().setUp()

    def tearDown(self) -> None:
        self.span_processor.shutdown()
        remove_baggage("__trulens_recording__")
        remove_baggage(SpanAttributes.RECORD_ID)
        for token in self.tokens[::-1]:
            context_api.detach(token)
        return super().tearDown()

    def test_get_func_name(self) -> None:
        self.assertEqual(
            get_func_name(lambda: None),
            "tests.unit.test_otel_instrument.TestOtelInstrument.test_get_func_name.<locals>.<lambda>",
        )
        self.assertEqual(
            get_func_name(self.test_get_func_name),
            "tests.unit.test_otel_instrument.TestOtelInstrument.test_get_func_name",
        )
        self.assertEqual(
            get_func_name(pd.DataFrame.transpose),
            "pandas.core.frame.DataFrame.transpose",
        )

    def test_sync_non_generator_function(self) -> None:
        # Set up instrumented function.
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.best_baby": ret
            }
        )
        def my_function():
            return "Kojikun"

        # Run the function.
        my_function()
        # Verify that the span is emitted correctly.
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].name,
            "tests.unit.test_otel_instrument.TestOtelInstrument.test_sync_non_generator_function.<locals>.my_function",
        )
        self.assertEqual(
            spans[0].attributes[f"{SpanAttributes.UNKNOWN.base}.best_baby"],
            "Kojikun",
        )

    def _test_sync_generator_function(
        self, my_function: Callable, test_name: str
    ) -> None:
        # Run the generator to completion.
        best_babies = my_function()
        for curr in best_babies:
            print(f"best_baby: {curr}")
        # Verify that the span is emitted correctly.
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].name,
            f"tests.unit.test_otel_instrument.TestOtelInstrument.{test_name}.<locals>.my_function",
        )
        self.assertTupleEqual(
            spans[0].attributes[f"{SpanAttributes.UNKNOWN.base}.best_babies"],
            ("Kojikun", "Nolan", "Sachiboy"),
        )
        # Run the generator partially.
        best_babies = my_function()
        for i, curr in enumerate(best_babies):
            print(f"best_baby: {curr}")
            if i == 1:
                break
        # Delete generator to ensure that the span is emitted.
        del best_babies
        gc.collect()
        # Verify that the span is emitted correctly.
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        self.assertEqual(
            spans[1].name,
            f"tests.unit.test_otel_instrument.TestOtelInstrument.{test_name}.<locals>.my_function",
        )
        self.assertTupleEqual(
            spans[1].attributes[f"{SpanAttributes.UNKNOWN.base}.best_babies"],
            ("Kojikun", "Nolan"),
        )

    def test_sync_generator_function(self) -> None:
        # Set up instrumented function.
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.best_babies": ret
            }
        )
        def my_function():
            yield "Kojikun"
            yield "Nolan"
            yield "Sachiboy"

        self._test_sync_generator_function(
            my_function, "test_sync_generator_function"
        )

    def test_sync_generator_passed_through_function(self) -> None:
        # Set up instrumented function.
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.best_babies": ret
            }
        )
        def my_function():
            return my_generator()

        def my_generator():
            yield "Kojikun"
            yield "Nolan"
            yield "Sachiboy"

        self._test_sync_generator_function(
            my_function, "test_sync_generator_passed_through_function"
        )

    def test_async_non_generator_function(self) -> None:
        # Set up instrumented function.
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.best_baby": ret
            }
        )
        async def my_function():
            await asyncio.sleep(0.00001)
            return "Kojikun"

        # Run the function.
        asyncio.run(my_function())
        # Verify that the span is emitted correctly.
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].name,
            "tests.unit.test_otel_instrument.TestOtelInstrument.test_async_non_generator_function.<locals>.my_function",
        )
        self.assertEqual(
            spans[0].attributes[f"{SpanAttributes.UNKNOWN.base}.best_baby"],
            "Kojikun",
        )

    def test_async_generator_function(self) -> None:
        # Set up instrumented function.
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.best_babies": ret
            }
        )
        async def my_function():
            await asyncio.sleep(0.00001)
            yield "Kojikun"
            yield "Nolan"
            yield "Sachiboy"

        # Helper to run the function.
        async def consume_async_generator(async_generator, num_iters):
            i = 0
            async for curr in async_generator:
                print(f"\t{curr}")
                i += 1
                if i == num_iters:
                    break

        # Run the generator to completion.
        asyncio.run(consume_async_generator(my_function(), 100))
        # Verify that the span is emitted correctly.
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].name,
            "tests.unit.test_otel_instrument.TestOtelInstrument.test_async_generator_function.<locals>.my_function",
        )
        self.assertTupleEqual(
            spans[0].attributes[f"{SpanAttributes.UNKNOWN.base}.best_babies"],
            ("Kojikun", "Nolan", "Sachiboy"),
        )
        # Run the generator partially.
        generator = my_function()
        asyncio.run(consume_async_generator(generator, 2))
        del generator
        gc.collect()
        # Verify that the span is emitted correctly.
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        self.assertEqual(
            spans[1].name,
            "tests.unit.test_otel_instrument.TestOtelInstrument.test_async_generator_function.<locals>.my_function",
        )
        self.assertTupleEqual(
            spans[1].attributes[f"{SpanAttributes.UNKNOWN.base}.best_babies"],
            ("Kojikun", "Nolan"),
        )

    def test_disabled_instrumentation(self) -> None:
        instrument_info = defaultdict(int)

        # Set up instrumented function.
        def fake_attributes(ret, exception, *args, **kwargs):
            instrument_info["cnt"] += 1
            instrument_info["ret"] = ret
            return {}

        @instrument(attributes=fake_attributes)
        def my_function() -> str:
            return "Kojikun is the best baby!"

        # Run the function.
        my_function()
        self.assertEqual(instrument_info["cnt"], 1)
        self.assertEqual(instrument_info["ret"], "Kojikun is the best baby!")
        instrument.disable_all_instrumentation()
        my_function()
        self.assertEqual(instrument_info["cnt"], 1)
        self.assertEqual(instrument_info["ret"], "Kojikun is the best baby!")
        instrument.enable_all_instrumentation()
        my_function()
        self.assertEqual(instrument_info["cnt"], 2)
        self.assertEqual(instrument_info["ret"], "Kojikun is the best baby!")

    def test_instrument_method_basic_third_party_class(self) -> None:
        class ThirdPartyRetriever:
            def retrieve(self, query: str) -> str:
                return f"result for {query}"

        instrument_method(ThirdPartyRetriever, "retrieve")

        result = ThirdPartyRetriever().retrieve("TruLens")

        self.assertEqual(result, "result for TruLens")
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertTrue(spans[0].name.endswith("ThirdPartyRetriever.retrieve"))

    def test_instrument_method_span_type_and_attribute_mapping(self) -> None:
        class ThirdPartyRetriever:
            def retrieve(self, query: str) -> list[str]:
                return [f"context for {query}"]

        instrument_method(
            ThirdPartyRetriever,
            "retrieve",
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            attributes={
                SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
                SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
            },
        )

        result = ThirdPartyRetriever().retrieve("What is TruLens?")

        self.assertEqual(result, ["context for What is TruLens?"])
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes[SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.RETRIEVAL,
        )
        self.assertEqual(
            spans[0].attributes[SpanAttributes.RETRIEVAL.QUERY_TEXT],
            "What is TruLens?",
        )
        self.assertTupleEqual(
            spans[0].attributes[SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS],
            ("context for What is TruLens?",),
        )

    def test_instrument_method_lambda_attributes(self) -> None:
        class ThirdPartyScorer:
            def score(self, text: str) -> int:
                return len(text)

        instrument_method(
            ThirdPartyScorer,
            "score",
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.score": ret,
                f"{SpanAttributes.UNKNOWN.base}.input_text": args[1],
            },
        )

        result = ThirdPartyScorer().score("abc")

        self.assertEqual(result, 3)
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes[f"{SpanAttributes.UNKNOWN.base}.score"],
            3,
        )
        self.assertEqual(
            spans[0].attributes[f"{SpanAttributes.UNKNOWN.base}.input_text"],
            "abc",
        )

    def test_instrument_method_twice_is_idempotent(self) -> None:
        class ThirdPartyClient:
            def call(self, value: str) -> str:
                return value.upper()

        instrument_method(ThirdPartyClient, "call")
        instrument_method(ThirdPartyClient, "call")

        result = ThirdPartyClient().call("trulens")

        self.assertEqual(result, "TRULENS")
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

    def test_attribute_mapped_to_defaulted_argument(self) -> None:
        top_k_key = f"{SpanAttributes.UNKNOWN.base}.top_k"

        @instrument(attributes={top_k_key: "top_k"})
        def retrieve(query: str, top_k: int = 3) -> list[str]:
            return [query] * top_k

        # Not passed: the default is recorded.
        self.assertEqual(retrieve("q"), ["q", "q", "q"])
        # Passed by keyword or position: the passed value wins.
        self.assertEqual(retrieve("q", top_k=1), ["q"])
        self.assertEqual(retrieve("q", 2), ["q", "q"])

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 3)
        self.assertEqual(
            [span.attributes[top_k_key] for span in spans], [3, 1, 2]
        )

    def test_async_attribute_mapped_to_defaulted_argument(self) -> None:
        top_k_key = f"{SpanAttributes.UNKNOWN.base}.top_k"

        @instrument(attributes={top_k_key: "top_k"})
        async def retrieve(query: str, top_k: int = 3) -> list[str]:
            await asyncio.sleep(0.00001)
            return [query] * top_k

        self.assertEqual(asyncio.run(retrieve("q")), ["q", "q", "q"])

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes[top_k_key], 3)

    def test_attribute_mapped_to_unknown_name_is_skipped(self) -> None:
        query_key = f"{SpanAttributes.UNKNOWN.base}.query"
        typo_key = f"{SpanAttributes.UNKNOWN.base}.top_k"

        @instrument(attributes={query_key: "query", typo_key: "topk"})
        def retrieve(query: str, top_k: int = 3) -> list[str]:
            return [query] * top_k

        with self.assertLogs(
            "trulens.core.otel.instrument", level="WARNING"
        ) as logs:
            result = retrieve("q")

        self.assertEqual(result, ["q", "q", "q"])
        self.assertTrue(
            any(typo_key in line and "topk" in line for line in logs.output)
        )
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes[query_key], "q")
        self.assertNotIn(typo_key, spans[0].attributes)

    def test_user_exception_still_propagates(self) -> None:
        @instrument(
            attributes={
                f"{SpanAttributes.UNKNOWN.base}.top_k": "top_k",
                f"{SpanAttributes.UNKNOWN.base}.typo": "topk",
            }
        )
        def retrieve(query: str, top_k: int = 3) -> list[str]:
            raise ValueError("retriever is down")

        with self.assertRaisesRegex(ValueError, "retriever is down"):
            retrieve("q")
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes[f"{SpanAttributes.UNKNOWN.base}.top_k"], 3
        )

    def test_span_group_three_hops_get_distinct_groups(self) -> None:
        """Three sequential span_group blocks produce three spans
        with distinct group labels — the core per-hop localization
        use case."""

        @instrument()
        def retrieve(query: str) -> str:
            return f"result for {query}"

        with span_group("hop1"):
            retrieve("q1")
        with span_group("hop2"):
            retrieve("q2")
        with span_group("hop3"):
            retrieve("q3")

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 3)
        # SPAN_GROUPS must be a tuple (OTel SDK converts list→tuple),
        # not a stringified list — compute_feedback_by_span_group
        # branches on isinstance(span_groups, str).
        for s in spans:
            self.assertIsInstance(
                s.attributes[SpanAttributes.SPAN_GROUPS], tuple
            )
        self.assertEqual(
            spans[0].attributes[SpanAttributes.SPAN_GROUPS], ("hop1",)
        )
        self.assertEqual(
            spans[1].attributes[SpanAttributes.SPAN_GROUPS], ("hop2",)
        )
        self.assertEqual(
            spans[2].attributes[SpanAttributes.SPAN_GROUPS], ("hop3",)
        )

    def test_span_group_nesting_merges(self) -> None:
        """Nested span_group() calls should merge group labels."""

        @instrument()
        def retrieve(query: str) -> str:
            return f"result for {query}"

        with span_group("hop1"):
            with span_group("retry"):
                retrieve("q")

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes[SpanAttributes.SPAN_GROUPS],
            ("hop1", "retry"),
        )

    def test_span_group_resets_on_exit(self) -> None:
        """After the with block exits, spans must not carry the group.
        Verifies the ContextVar token is properly reset."""

        @instrument()
        def retrieve(query: str) -> str:
            return f"result for {query}"

        with span_group("hop1"):
            retrieve("q1")
        # This span is created OUTSIDE the span_group block.
        retrieve("q2")

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        self.assertEqual(
            spans[0].attributes[SpanAttributes.SPAN_GROUPS], ("hop1",)
        )
        self.assertNotIn(SpanAttributes.SPAN_GROUPS, spans[1].attributes)

    def test_span_group_resets_after_exception(self) -> None:
        """If the body of a span_group block raises, the group must
        still be cleared — no leaking into downstream spans."""

        @instrument()
        def retrieve(query: str) -> str:
            return f"result for {query}"

        with self.assertRaises(ValueError):
            with span_group("x"):
                raise ValueError("boom")

        retrieve("after_exception")

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertNotIn(SpanAttributes.SPAN_GROUPS, spans[0].attributes)

    def test_span_group_propagates_through_nested_calls(self) -> None:
        """span_group() must tag all spans inside the block,
        including nested instrumented calls multiple stack frames
        deep — without threading an argument."""

        @instrument()
        def leaf() -> str:
            return "leaf"

        @instrument()
        def middle() -> str:
            return leaf()

        @instrument()
        def top() -> str:
            return middle()

        with span_group("group_a"):
            top()

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 3)
        for s in spans:
            self.assertEqual(
                s.attributes[SpanAttributes.SPAN_GROUPS], ("group_a",)
            )

    def test_generation_span_has_client_span_kind(self) -> None:
        """GENERATION spans should have SpanKind.CLIENT (outbound LLM calls)."""

        @instrument(span_type=SpanAttributes.SpanType.GENERATION)
        def call_llm(prompt: str) -> str:
            return "response"

        call_llm("hello")

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].kind, SpanKind.CLIENT)

    def test_non_generation_span_has_internal_span_kind(self) -> None:
        """Non-GENERATION spans should have SpanKind.INTERNAL."""

        @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
        def retrieve(query: str) -> list:
            return ["doc1"]

        @instrument()
        def plain_func() -> str:
            return "ok"

        retrieve("test")
        plain_func()

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].kind, SpanKind.INTERNAL)
        self.assertEqual(spans[1].kind, SpanKind.INTERNAL)

    def test_generation_span_exports_as_client_kind_in_proto(self) -> None:
        """A GENERATION span should convert to SPAN_KIND_CLIENT in proto."""
        from opentelemetry.proto.trace.v1.trace_pb2 import Span as SpanProto
        from trulens.experimental.otel_tracing.core.exporter.utils import (
            convert_readable_span_to_proto,
        )

        @instrument(span_type=SpanAttributes.SpanType.GENERATION)
        def call_llm(prompt: str) -> str:
            return "response"

        call_llm("hello")

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        proto = convert_readable_span_to_proto(spans[0])
        self.assertEqual(proto.kind, SpanProto.SpanKind.SPAN_KIND_CLIENT)


class _StubApp:
    """Just enough of an app for record root spans to be finalized."""

    def main_input(self, func, sig, bindings):
        return str(next(iter(bindings.arguments.values()), None))

    def main_output(self, func, sig, bindings, ret):
        return str(ret)


class TestOtelInstrumentGeneratorContext(unittest.TestCase):
    """A stream's span context must stay out of the caller between chunks.

    Unlike `TestOtelInstrument`, no record id is set up front, so every top
    level instrumented call starts a record of its own, as it does in an app.
    """

    @classmethod
    def setUpClass(cls) -> None:
        instrument.enable_all_instrumentation()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        instrument.disable_all_instrumentation()
        return super().tearDownClass()

    def setUp(self) -> None:
        self.exporter = InMemorySpanExporter()
        _set_up_tracer_provider()
        self.span_processor = SimpleSpanProcessor(self.exporter)
        trace.get_tracer_provider().add_span_processor(self.span_processor)
        self.recording = Recording(None)
        context = set_baggage("__trulens_recording__", self.recording)
        context = set_baggage("__trulens_app__", _StubApp(), context=context)
        self.token = context_api.attach(context)
        return super().setUp()

    def tearDown(self) -> None:
        self.span_processor.shutdown()
        context_api.detach(self.token)
        return super().tearDown()

    def _spans_by_name(self) -> dict:
        return {
            span.name.rsplit(".", 1)[-1]: span
            for span in self.exporter.get_finished_spans()
        }

    def _assert_nothing_attached(self) -> None:
        self.assertIsNone(baggage.get_baggage(SpanAttributes.RECORD_ID))
        self.assertFalse(trace.get_current_span().get_span_context().is_valid)

    def _assert_separate_records(self, stream_span, answer_span) -> None:
        self.assertEqual(len(self.recording), 2)
        self.assertIsNone(answer_span.parent)
        self.assertEqual(
            answer_span.attributes[SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.RECORD_ROOT,
        )
        self.assertNotEqual(
            answer_span.attributes[SpanAttributes.RECORD_ID],
            stream_span.attributes[SpanAttributes.RECORD_ID],
        )

    def test_sync_stream_abandoned_after_first_chunk(self) -> None:
        @instrument()
        def stream(question: str):
            yield "Kojikun"
            yield "Nolan"

        @instrument()
        def answer(question: str) -> str:
            return "Sachiboy"

        with self.assertNoLogs("opentelemetry.context", level="ERROR"):
            chunks = stream("first")
            self.assertEqual(next(chunks), "Kojikun")
            self._assert_nothing_attached()
            answer("second")
            del chunks
            gc.collect()

        self._assert_nothing_attached()
        spans = self._spans_by_name()
        self._assert_separate_records(spans["stream"], spans["answer"])
        self.assertEqual(
            spans["stream"].attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
            "['Kojikun']",
        )

    def test_async_stream_abandoned_after_first_chunk(self) -> None:
        @instrument()
        async def stream(question: str):
            yield "Kojikun"
            yield "Nolan"

        @instrument()
        async def answer(question: str) -> str:
            return "Sachiboy"

        async def run():
            async for chunk in stream("first"):
                self.assertEqual(chunk, "Kojikun")
                break
            self._assert_nothing_attached()
            await answer("second")

        with self.assertNoLogs("opentelemetry.context", level="ERROR"):
            asyncio.run(run())
            gc.collect()

        self._assert_nothing_attached()
        spans = self._spans_by_name()
        self._assert_separate_records(spans["stream"], spans["answer"])

    def test_sync_stream_keeps_nested_calls_as_children(self) -> None:
        @instrument()
        def inner(name: str) -> str:
            return name.upper()

        @instrument()
        def stream(question: str):
            yield inner("Kojikun")
            yield inner("Nolan")

        self.assertEqual(list(stream("first")), ["KOJIKUN", "NOLAN"])

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 3)
        stream_span = spans[-1]
        self.assertEqual(len(self.recording), 1)
        self.assertEqual(
            stream_span.attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
            "['KOJIKUN', 'NOLAN']",
        )
        self.assertEqual(
            stream_span.attributes[SpanAttributes.GENERATION.CHUNKS_RECEIVED],
            2,
        )
        for inner_span in spans[:2]:
            self.assertEqual(
                inner_span.parent.span_id, stream_span.context.span_id
            )
            self.assertEqual(
                inner_span.attributes[SpanAttributes.RECORD_ID],
                stream_span.attributes[SpanAttributes.RECORD_ID],
            )

    def test_async_stream_keeps_nested_calls_as_children(self) -> None:
        @instrument()
        async def inner(name: str) -> str:
            return name.upper()

        @instrument()
        async def stream(question: str):
            yield await inner("Kojikun")
            yield await inner("Nolan")

        async def run():
            return [chunk async for chunk in stream("first")]

        self.assertEqual(asyncio.run(run()), ["KOJIKUN", "NOLAN"])

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 3)
        stream_span = spans[-1]
        self.assertEqual(len(self.recording), 1)
        self.assertEqual(
            stream_span.attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
            "['KOJIKUN', 'NOLAN']",
        )
        self.assertEqual(
            stream_span.attributes[SpanAttributes.GENERATION.CHUNKS_RECEIVED],
            2,
        )
        for inner_span in spans[:2]:
            self.assertEqual(
                inner_span.parent.span_id, stream_span.context.span_id
            )

    def test_sync_stream_error_marks_span(self) -> None:
        @instrument()
        def stream(question: str):
            yield "Kojikun"
            raise ValueError("no more babies")

        with self.assertRaises(ValueError):
            list(stream("first"))

        self._assert_nothing_attached()
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, trace.StatusCode.ERROR)
        self.assertEqual(
            spans[0].attributes[SpanAttributes.CALL.ERROR], "no more babies"
        )

    def test_async_stream_error_marks_span(self) -> None:
        @instrument()
        async def stream(question: str):
            yield "Kojikun"
            raise ValueError("no more babies")

        async def run():
            return [chunk async for chunk in stream("first")]

        with self.assertRaises(ValueError):
            asyncio.run(run())

        self._assert_nothing_attached()
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, trace.StatusCode.ERROR)
        self.assertEqual(
            spans[0].attributes[SpanAttributes.CALL.ERROR], "no more babies"
        )


def _passthrough(func):
    """A plain sync decorator, shaped like OpenAI's `required_args`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _run_to_completion(func):
    """A sync decorator that runs the coroutine itself."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class TestOtelInstrumentAsyncBehindDecorator(unittest.TestCase):
    """An `async def` behind a sync decorator is traced like an `async def`.

    As in `TestOtelInstrumentGeneratorContext`, every top level instrumented
    call starts a record of its own.
    """

    @classmethod
    def setUpClass(cls) -> None:
        instrument.enable_all_instrumentation()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        instrument.disable_all_instrumentation()
        return super().tearDownClass()

    def setUp(self) -> None:
        self.exporter = InMemorySpanExporter()
        _set_up_tracer_provider()
        self.span_processor = SimpleSpanProcessor(self.exporter)
        trace.get_tracer_provider().add_span_processor(self.span_processor)
        self.recording = Recording(None)
        context = set_baggage("__trulens_recording__", self.recording)
        context = set_baggage("__trulens_app__", _StubApp(), context=context)
        self.token = context_api.attach(context)
        return super().setUp()

    def tearDown(self) -> None:
        self.span_processor.shutdown()
        context_api.detach(self.token)
        return super().tearDown()

    def _spans_by_name(self) -> dict:
        return {
            span.name.rsplit(".", 1)[-1]: span
            for span in self.exporter.get_finished_spans()
        }

    def _assert_nested_child(self, outer_span, inner_span) -> None:
        self.assertEqual(len(self.recording), 1)
        self.assertEqual(
            outer_span.attributes[SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.RECORD_ROOT,
        )
        self.assertEqual(inner_span.parent.span_id, outer_span.context.span_id)
        self.assertEqual(
            inner_span.attributes[SpanAttributes.SPAN_TYPE],
            SpanAttributes.SpanType.RETRIEVAL,
        )
        self.assertEqual(
            inner_span.attributes[SpanAttributes.RECORD_ID],
            outer_span.attributes[SpanAttributes.RECORD_ID],
        )

    def test_async_method_behind_sync_decorator(self) -> None:
        @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
        async def retrieve(query: str) -> list:
            return ["Kojikun"]

        class App:
            @instrument()
            @_passthrough
            async def query(self, question: str) -> str:
                await asyncio.sleep(0.05)
                return str(await retrieve(question))

        self.assertEqual(asyncio.run(App().query("q")), "['Kojikun']")

        self.assertIsNone(baggage.get_baggage(SpanAttributes.RECORD_ID))
        spans = self._spans_by_name()
        self.assertEqual(len(spans), 2)
        self._assert_nested_child(spans["query"], spans["retrieve"])
        self.assertEqual(
            spans["query"].attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
            "['Kojikun']",
        )
        self.assertGreaterEqual(
            spans["query"].end_time - spans["query"].start_time, 50_000_000
        )

    def test_async_behind_sync_decorator_error_marks_span(self) -> None:
        @instrument()
        @_passthrough
        async def query(question: str) -> str:
            await asyncio.sleep(0.00001)
            raise ValueError("no more babies")

        with self.assertRaises(ValueError):
            asyncio.run(query("q"))

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, trace.StatusCode.ERROR)
        self.assertEqual(
            spans[0].attributes[SpanAttributes.CALL.ERROR], "no more babies"
        )

    def test_async_behind_sync_decorator_cancelled_ends_span(self) -> None:
        @instrument()
        @_passthrough
        async def query(question: str) -> str:
            await asyncio.sleep(10)
            return "Kojikun"

        async def run():
            task = asyncio.ensure_future(query("q"))
            await asyncio.sleep(0.01)
            task.cancel()
            await task

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run())

        self.assertIsNone(baggage.get_baggage(SpanAttributes.RECORD_ID))
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(len(self.recording), 1)

    def test_sync_decorator_that_runs_the_coroutine(self) -> None:
        @instrument()
        @_run_to_completion
        async def query(question: str) -> str:
            await asyncio.sleep(0.00001)
            return "Kojikun"

        self.assertEqual(query("q"), "Kojikun")

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes[SpanAttributes.RECORD_ROOT.OUTPUT], "Kojikun"
        )

    def test_plain_async_function_unchanged(self) -> None:
        @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
        async def retrieve(query: str) -> list:
            return ["Kojikun"]

        @instrument()
        async def query(question: str) -> str:
            await asyncio.sleep(0.00001)
            return str(await retrieve(question))

        self.assertEqual(asyncio.run(query("q")), "['Kojikun']")

        spans = self._spans_by_name()
        self._assert_nested_child(spans["query"], spans["retrieve"])
        self.assertEqual(
            spans["query"].attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
            "['Kojikun']",
        )

    def test_plain_sync_function_unchanged(self) -> None:
        @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
        def retrieve(query: str) -> list:
            return ["Kojikun"]

        @instrument()
        def query(question: str) -> str:
            return str(retrieve(question))

        self.assertEqual(query("q"), "['Kojikun']")

        spans = self._spans_by_name()
        self._assert_nested_child(spans["query"], spans["retrieve"])
        self.assertEqual(
            spans["query"].attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
            "['Kojikun']",
        )

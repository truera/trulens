import asyncio
from collections import defaultdict
import gc
from typing import Callable
import unittest

from opentelemetry import trace
from opentelemetry.baggage import remove_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
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

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

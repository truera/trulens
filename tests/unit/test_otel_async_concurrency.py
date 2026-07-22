import asyncio
import builtins
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from inspect import BoundArguments
from inspect import Signature
import sys
from typing import Any, cast
import unittest

from opentelemetry import trace
from opentelemetry.baggage import get_baggage
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanContext
import pytest
from trulens.core.otel.instrument import OtelRecordingContext
from trulens.core.otel.instrument import instrument
from trulens.experimental.otel_tracing.core.session import (
    _set_up_tracer_provider,
)
from trulens.otel.semconv.trace import SpanAttributes

_PARENT_SPAN_NAME = "issue_2441_parent"
_CHILD_A_SPAN_NAME = "issue_2441_child_a"
_CHILD_B_SPAN_NAME = "issue_2441_child_b"
_SECOND_ROOT_SPAN_NAME = "issue_2441_second_root"
_CANCEL_PARENT_SPAN_NAME = "issue_2441_cancel_parent"
_CANCELLED_CHILD_SPAN_NAME = "issue_2441_cancelled_child"
_POST_CANCEL_ROOT_SPAN_NAME = "issue_2441_post_cancel_root"
_EXCEPTION_GROUP_PARENT_SPAN_NAME = "issue_2441_exception_group_parent"
_FAILING_CHILD_SPAN_NAME = "issue_2441_failing_child"
_CANCELLED_SIBLING_SPAN_NAME = "issue_2441_cancelled_sibling"
_POST_EXCEPTION_GROUP_ROOT_SPAN_NAME = "issue_2441_post_exception_group_root"
_RECORDING_BAGGAGE_KEY = "__trulens_recording__"


@dataclass(frozen=True)
class _ContextSnapshot:
    span_context: SpanContext
    record_id: Any
    recording: Any


@dataclass(frozen=True)
class _CancellationResult:
    context_before: _ContextSnapshot
    context_after_cancellation: _ContextSnapshot
    context_after_new_recording: _ContextSnapshot
    caught_exception: BaseException | None
    child_cancelled_error: asyncio.CancelledError | None
    parent_span_id_after_cancellation: int | None


@dataclass(frozen=True)
class _TaskGroupCancellationResult:
    context_before: _ContextSnapshot
    context_after_exception_group: _ContextSnapshot
    context_after_new_recording: _ContextSnapshot
    caught_exception: Exception | None
    expected_error_type: type[Exception]
    cancelled_sibling_error: asyncio.CancelledError | None
    cancelled_sibling_span_ids: list[int]


class _TestApp:
    def main_input(
        self,
        _func: Callable,
        _sig: Signature,
        _bindings: BoundArguments,
    ) -> str:
        return "issue_2441_input"

    def main_output(
        self,
        _func: Callable,
        _sig: Signature,
        _bindings: BoundArguments,
        ret: Any,
    ) -> str:
        return str(ret)


class TestOtelAsyncConcurrency(unittest.TestCase):
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
        self.exporter.clear()
        self.test_app = _TestApp()
        tracer_provider = _set_up_tracer_provider()
        self.span_processor = SimpleSpanProcessor(self.exporter)
        tracer_provider.add_span_processor(self.span_processor)
        return super().setUp()

    def tearDown(self) -> None:
        self.span_processor.shutdown()
        self.exporter.clear()
        return super().tearDown()

    def _recording_context(self) -> OtelRecordingContext:
        return OtelRecordingContext(
            tru_app=cast(Any, self.test_app),
            app_name="Issue2441TestApp",
            app_version="v1",
            run_name="issue_2441_run",
            input_id="issue_2441_input",
        )

    @staticmethod
    def _span_context(span: ReadableSpan) -> SpanContext:
        span_context = span.get_span_context()
        if span_context is None:
            raise AssertionError(f"Span {span.name!r} has no span context")
        return span_context

    @staticmethod
    def _context_snapshot() -> _ContextSnapshot:
        return _ContextSnapshot(
            span_context=trace.get_current_span().get_span_context(),
            record_id=get_baggage(SpanAttributes.RECORD_ID),
            recording=get_baggage(_RECORDING_BAGGAGE_KEY),
        )

    def _assert_context_restored(
        self,
        expected: _ContextSnapshot,
        actual: _ContextSnapshot,
    ) -> None:
        self.assertEqual(actual.span_context, expected.span_context)
        self.assertEqual(actual.record_id, expected.record_id)
        self.assertIs(actual.recording, expected.recording)

    async def _run_concurrent_children(
        self, *, use_task_group: bool
    ) -> tuple[bool, dict[str, list[int]]]:
        entered_count = 0
        entered_lock = asyncio.Lock()
        all_entered = asyncio.Event()
        release = asyncio.Event()
        observations: dict[str, list[int]] = {
            "a": [],
            "b": [],
        }

        async def wait_at_barrier(branch: str) -> str:
            nonlocal entered_count

            observations[branch].append(
                trace.get_current_span().get_span_context().span_id
            )
            async with entered_lock:
                entered_count += 1
                if entered_count == 2:
                    all_entered.set()

            await release.wait()
            observations[branch].append(
                trace.get_current_span().get_span_context().span_id
            )
            return branch

        @instrument(name=_CHILD_A_SPAN_NAME)
        async def child_a() -> str:
            return await wait_at_barrier("a")

        @instrument(name=_CHILD_B_SPAN_NAME)
        async def child_b() -> str:
            return await wait_at_barrier("b")

        overlap_observed = False

        @instrument(name=_PARENT_SPAN_NAME)
        async def parent() -> tuple[str, str]:
            nonlocal overlap_observed

            if use_task_group:
                async with asyncio.TaskGroup() as task_group:
                    task_a = task_group.create_task(child_a())
                    task_b = task_group.create_task(child_b())
                    await all_entered.wait()
                    overlap_observed = entered_count == 2
                    release.set()
                return task_a.result(), task_b.result()

            pending = asyncio.gather(child_a(), child_b())
            await all_entered.wait()
            overlap_observed = entered_count == 2
            release.set()
            result_a, result_b = await pending
            return result_a, result_b

        with self._recording_context():
            self.assertEqual(await parent(), ("a", "b"))

        return overlap_observed, observations

    def _target_spans(
        self, target_names: set[str] | None = None
    ) -> dict[str, ReadableSpan]:
        if target_names is None:
            target_names = {
                _PARENT_SPAN_NAME,
                _CHILD_A_SPAN_NAME,
                _CHILD_B_SPAN_NAME,
            }
        target_spans = [
            span
            for span in self.exporter.get_finished_spans()
            if span.name in target_names
        ]
        name_counts = Counter(span.name for span in target_spans)

        self.assertEqual(
            name_counts,
            Counter({name: 1 for name in target_names}),
        )
        return {span.name: span for span in target_spans}

    def _assert_sibling_structure(
        self, spans_by_name: dict[str, ReadableSpan]
    ) -> None:
        parent = spans_by_name[_PARENT_SPAN_NAME]
        child_a = spans_by_name[_CHILD_A_SPAN_NAME]
        child_b = spans_by_name[_CHILD_B_SPAN_NAME]
        parent_context = self._span_context(parent)
        child_a_context = self._span_context(child_a)
        child_b_context = self._span_context(child_b)

        self.assertEqual(
            {
                parent_context.trace_id,
                child_a_context.trace_id,
                child_b_context.trace_id,
            },
            {parent_context.trace_id},
        )
        child_a_parent = child_a.parent
        child_b_parent = child_b.parent
        self.assertIsNotNone(child_a_parent)
        self.assertIsNotNone(child_b_parent)
        assert child_a_parent is not None
        assert child_b_parent is not None
        self.assertEqual(child_a_parent.span_id, parent_context.span_id)
        self.assertEqual(child_b_parent.span_id, parent_context.span_id)
        self.assertNotEqual(child_a_context.span_id, child_b_context.span_id)
        self.assertNotEqual(child_a_parent.span_id, child_b_context.span_id)
        self.assertNotEqual(child_b_parent.span_id, child_a_context.span_id)

        for span in spans_by_name.values():
            self.assertIsNotNone(span.end_time)

    def _run_cancelled_child_scenario(
        self, *, attributes: Any = None
    ) -> _CancellationResult:
        context_before = self._context_snapshot()
        context_after_cancellation = context_before
        context_after_new_recording = context_before
        child_started = asyncio.Event()
        wait_for_cancellation = asyncio.Event()
        caught_exception: BaseException | None = None
        child_cancelled_error: asyncio.CancelledError | None = None
        parent_span_id_after_cancellation: int | None = None

        @instrument(name=_CANCELLED_CHILD_SPAN_NAME, attributes=attributes)
        async def cancelled_child() -> None:
            nonlocal child_cancelled_error
            child_started.set()
            try:
                await wait_for_cancellation.wait()
            except asyncio.CancelledError as error:
                child_cancelled_error = error
                raise

        @instrument(name=_CANCEL_PARENT_SPAN_NAME)
        async def cancel_parent() -> None:
            nonlocal caught_exception
            nonlocal parent_span_id_after_cancellation

            task = asyncio.create_task(cancelled_child())
            await child_started.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError as error:
                caught_exception = error
            except ValueError as error:
                caught_exception = error
            parent_span_id_after_cancellation = (
                trace.get_current_span().get_span_context().span_id
            )

        @instrument(name=_POST_CANCEL_ROOT_SPAN_NAME)
        async def post_cancel_root() -> None:
            return None

        async def run_scenario() -> None:
            nonlocal context_after_cancellation
            nonlocal context_after_new_recording

            with self._recording_context():
                await cancel_parent()
            context_after_cancellation = self._context_snapshot()

            with self._recording_context():
                await post_cancel_root()
            context_after_new_recording = self._context_snapshot()

        asyncio.run(run_scenario())
        return _CancellationResult(
            context_before=context_before,
            context_after_cancellation=context_after_cancellation,
            context_after_new_recording=context_after_new_recording,
            caught_exception=caught_exception,
            child_cancelled_error=child_cancelled_error,
            parent_span_id_after_cancellation=(
                parent_span_id_after_cancellation
            ),
        )

    def _run_task_group_cancellation_scenario(
        self,
    ) -> _TaskGroupCancellationResult:
        class ExpectedTaskGroupError(RuntimeError):
            pass

        context_before = self._context_snapshot()
        context_after_exception_group = context_before
        context_after_new_recording = context_before
        entered_count = 0
        entered_lock = asyncio.Lock()
        both_entered = asyncio.Event()
        wait_for_cancellation = asyncio.Event()
        caught_exception: Exception | None = None
        cancelled_sibling_error: asyncio.CancelledError | None = None
        cancelled_sibling_span_ids: list[int] = []

        async def enter_barrier() -> None:
            nonlocal entered_count
            async with entered_lock:
                entered_count += 1
                if entered_count == 2:
                    both_entered.set()
            await both_entered.wait()

        @instrument(name=_FAILING_CHILD_SPAN_NAME)
        async def failing_child() -> None:
            await enter_barrier()
            raise ExpectedTaskGroupError("expected TaskGroup child failure")

        @instrument(name=_CANCELLED_SIBLING_SPAN_NAME)
        async def cancelled_sibling() -> None:
            nonlocal cancelled_sibling_error
            await enter_barrier()
            cancelled_sibling_span_ids.append(
                trace.get_current_span().get_span_context().span_id
            )
            try:
                await wait_for_cancellation.wait()
            except asyncio.CancelledError as error:
                cancelled_sibling_error = error
                raise

        @instrument(name=_EXCEPTION_GROUP_PARENT_SPAN_NAME)
        async def exception_group_parent() -> None:
            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(failing_child())
                task_group.create_task(cancelled_sibling())

        @instrument(name=_POST_EXCEPTION_GROUP_ROOT_SPAN_NAME)
        async def post_exception_group_root() -> None:
            return None

        async def run_scenario() -> None:
            nonlocal caught_exception
            nonlocal context_after_exception_group
            nonlocal context_after_new_recording

            try:
                with self._recording_context():
                    await exception_group_parent()
            except builtins.ExceptionGroup as error:
                caught_exception = error
            context_after_exception_group = self._context_snapshot()

            with self._recording_context():
                await post_exception_group_root()
            context_after_new_recording = self._context_snapshot()

        asyncio.run(run_scenario())
        return _TaskGroupCancellationResult(
            context_before=context_before,
            context_after_exception_group=context_after_exception_group,
            context_after_new_recording=context_after_new_recording,
            caught_exception=caught_exception,
            expected_error_type=ExpectedTaskGroupError,
            cancelled_sibling_error=cancelled_sibling_error,
            cancelled_sibling_span_ids=cancelled_sibling_span_ids,
        )

    def test_asyncio_gather_creates_sibling_spans_under_instrumented_parent(
        self,
    ) -> None:
        overlap_observed, _ = asyncio.run(
            self._run_concurrent_children(use_task_group=False)
        )

        self.assertTrue(overlap_observed)
        self._assert_sibling_structure(self._target_spans())

    def test_asyncio_gather_does_not_leak_active_child_span_between_branches(
        self,
    ) -> None:
        overlap_observed, observations = asyncio.run(
            self._run_concurrent_children(use_task_group=False)
        )
        spans_by_name = self._target_spans()
        parent_context = self._span_context(spans_by_name[_PARENT_SPAN_NAME])
        child_a_context = self._span_context(spans_by_name[_CHILD_A_SPAN_NAME])
        child_b_context = self._span_context(spans_by_name[_CHILD_B_SPAN_NAME])

        self.assertTrue(overlap_observed)
        self.assertEqual(observations["a"], [child_a_context.span_id] * 2)
        self.assertEqual(observations["b"], [child_b_context.span_id] * 2)
        self.assertNotEqual(observations["a"][0], observations["b"][0])
        self.assertNotEqual(observations["a"][0], parent_context.span_id)
        self.assertNotEqual(observations["b"][0], parent_context.span_id)

    @pytest.mark.skipif(
        sys.version_info < (3, 11),
        reason="asyncio.TaskGroup requires Python 3.11+",
    )
    def test_task_group_creates_sibling_spans_under_instrumented_parent(
        self,
    ) -> None:
        overlap_observed, _ = asyncio.run(
            self._run_concurrent_children(use_task_group=True)
        )

        self.assertTrue(overlap_observed)
        self._assert_sibling_structure(self._target_spans())

    def test_async_concurrency_restores_otel_and_trulens_context(self) -> None:
        initial_span_context = trace.get_current_span().get_span_context()
        initial_record_id = get_baggage(SpanAttributes.RECORD_ID)
        initial_recording = get_baggage(_RECORDING_BAGGAGE_KEY)

        async def run_two_recordings() -> None:
            await self._run_concurrent_children(use_task_group=False)

            self.assertEqual(
                trace.get_current_span().get_span_context(),
                initial_span_context,
            )
            self.assertEqual(
                get_baggage(SpanAttributes.RECORD_ID), initial_record_id
            )
            self.assertIs(
                get_baggage(_RECORDING_BAGGAGE_KEY), initial_recording
            )

            @instrument(name=_SECOND_ROOT_SPAN_NAME)
            async def second_root() -> None:
                return None

            with self._recording_context():
                await second_root()

        asyncio.run(run_two_recordings())

        spans_by_name = self._target_spans()
        first_span_ids = {
            self._span_context(span).span_id for span in spans_by_name.values()
        }
        first_trace_id = self._span_context(
            spans_by_name[_PARENT_SPAN_NAME]
        ).trace_id
        second_roots = [
            span
            for span in self.exporter.get_finished_spans()
            if span.name == _SECOND_ROOT_SPAN_NAME
        ]

        self.assertEqual(len(second_roots), 1)
        second_root = second_roots[0]
        second_root_context = self._span_context(second_root)
        self.assertNotEqual(second_root_context.trace_id, first_trace_id)
        self.assertIsNone(second_root.parent)
        self.assertNotIn(second_root_context.span_id, first_span_ids)
        self.assertIsNotNone(second_root.end_time)
        self.assertEqual(
            trace.get_current_span().get_span_context(), initial_span_context
        )
        self.assertEqual(
            get_baggage(SpanAttributes.RECORD_ID), initial_record_id
        )
        self.assertIs(get_baggage(_RECORDING_BAGGAGE_KEY), initial_recording)

    def test_cancelled_instrumented_task_restores_context_and_finishes_span(
        self,
    ) -> None:
        result = self._run_cancelled_child_scenario()

        spans_by_name = self._target_spans({
            _CANCEL_PARENT_SPAN_NAME,
            _CANCELLED_CHILD_SPAN_NAME,
            _POST_CANCEL_ROOT_SPAN_NAME,
        })
        parent = spans_by_name[_CANCEL_PARENT_SPAN_NAME]
        child = spans_by_name[_CANCELLED_CHILD_SPAN_NAME]
        post_cancel_root_span = spans_by_name[_POST_CANCEL_ROOT_SPAN_NAME]
        parent_context = self._span_context(parent)
        child_context = self._span_context(child)
        post_cancel_context = self._span_context(post_cancel_root_span)
        child_parent = child.parent

        self.assertIsNotNone(child_parent)
        assert child_parent is not None
        self.assertEqual(child_parent.span_id, parent_context.span_id)
        self.assertEqual(child_context.trace_id, parent_context.trace_id)
        self.assertIsInstance(result.caught_exception, asyncio.CancelledError)
        self.assertIs(result.caught_exception, result.child_cancelled_error)
        self.assertEqual(
            result.parent_span_id_after_cancellation, parent_context.span_id
        )
        self.assertIsNotNone(child.end_time)
        self.assertIsNotNone(parent.end_time)
        self.assertIsNone(post_cancel_root_span.parent)
        self.assertNotEqual(
            post_cancel_context.trace_id, parent_context.trace_id
        )
        self.assertIsNotNone(post_cancel_root_span.end_time)
        self._assert_context_restored(
            result.context_before,
            result.context_after_cancellation,
        )
        self._assert_context_restored(
            result.context_before,
            result.context_after_new_recording,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 11),
        reason="asyncio.TaskGroup requires Python 3.11+",
    )
    def test_task_group_exception_group_cancellation_keeps_span_isolation(
        self,
    ) -> None:
        result = self._run_task_group_cancellation_scenario()
        exception_group_type = builtins.ExceptionGroup

        self.assertIsInstance(result.caught_exception, exception_group_type)
        assert result.caught_exception is not None
        exception_group = cast(Any, result.caught_exception)
        self.assertTrue(
            any(
                isinstance(exception, result.expected_error_type)
                for exception in exception_group.exceptions
            )
        )
        self.assertIsInstance(
            result.cancelled_sibling_error, asyncio.CancelledError
        )

        spans_by_name = self._target_spans({
            _EXCEPTION_GROUP_PARENT_SPAN_NAME,
            _FAILING_CHILD_SPAN_NAME,
            _CANCELLED_SIBLING_SPAN_NAME,
            _POST_EXCEPTION_GROUP_ROOT_SPAN_NAME,
        })
        parent = spans_by_name[_EXCEPTION_GROUP_PARENT_SPAN_NAME]
        failing_child_span = spans_by_name[_FAILING_CHILD_SPAN_NAME]
        cancelled_sibling_span = spans_by_name[_CANCELLED_SIBLING_SPAN_NAME]
        post_exception_root = spans_by_name[
            _POST_EXCEPTION_GROUP_ROOT_SPAN_NAME
        ]
        parent_context = self._span_context(parent)
        failing_context = self._span_context(failing_child_span)
        cancelled_context = self._span_context(cancelled_sibling_span)
        post_exception_context = self._span_context(post_exception_root)
        failing_parent = failing_child_span.parent
        cancelled_parent = cancelled_sibling_span.parent

        self.assertIsNotNone(failing_parent)
        self.assertIsNotNone(cancelled_parent)
        assert failing_parent is not None
        assert cancelled_parent is not None
        self.assertEqual(failing_parent.span_id, parent_context.span_id)
        self.assertEqual(cancelled_parent.span_id, parent_context.span_id)
        self.assertEqual(failing_context.trace_id, parent_context.trace_id)
        self.assertEqual(cancelled_context.trace_id, parent_context.trace_id)
        self.assertNotEqual(failing_context.span_id, cancelled_context.span_id)
        self.assertNotEqual(failing_parent.span_id, cancelled_context.span_id)
        self.assertNotEqual(cancelled_parent.span_id, failing_context.span_id)
        self.assertEqual(
            result.cancelled_sibling_span_ids, [cancelled_context.span_id]
        )
        self.assertIsNotNone(failing_child_span.end_time)
        self.assertIsNotNone(cancelled_sibling_span.end_time)
        self.assertIsNotNone(parent.end_time)
        self.assertIsNone(post_exception_root.parent)
        self.assertNotEqual(
            post_exception_context.trace_id, parent_context.trace_id
        )
        self.assertIsNotNone(post_exception_root.end_time)
        self._assert_context_restored(
            result.context_before,
            result.context_after_exception_group,
        )
        self._assert_context_restored(
            result.context_before,
            result.context_after_new_recording,
        )

    def test_cancelled_instrumented_task_records_function_metadata(
        self,
    ) -> None:
        result = self._run_cancelled_child_scenario()

        spans_by_name = self._target_spans({
            _CANCEL_PARENT_SPAN_NAME,
            _CANCELLED_CHILD_SPAN_NAME,
        })
        parent = spans_by_name[_CANCEL_PARENT_SPAN_NAME]
        child = spans_by_name[_CANCELLED_CHILD_SPAN_NAME]
        parent_context = self._span_context(parent)
        child_context = self._span_context(child)
        child_parent = child.parent
        child_attributes = child.attributes or {}

        self.assertIsNotNone(child_parent)
        assert child_parent is not None
        self.assertEqual(child_parent.span_id, parent_context.span_id)
        self.assertEqual(child_context.trace_id, parent_context.trace_id)
        self.assertIsInstance(result.caught_exception, asyncio.CancelledError)
        self.assertIs(result.caught_exception, result.child_cancelled_error)
        self.assertIsNotNone(child.end_time)
        self.assertIn(SpanAttributes.CALL.FUNCTION, child_attributes)
        self.assertNotIn(SpanAttributes.CALL.RETURN, child_attributes)
        self.assertNotIn(SpanAttributes.CALL.ERROR, child_attributes)
        self._assert_context_restored(
            result.context_before,
            result.context_after_cancellation,
        )
        self._assert_context_restored(
            result.context_before,
            result.context_after_new_recording,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 11),
        reason="asyncio.TaskGroup requires Python 3.11+",
    )
    def test_task_group_cancelled_sibling_records_finalization_metadata(
        self,
    ) -> None:
        result = self._run_task_group_cancellation_scenario()
        exception_group_type = builtins.ExceptionGroup

        self.assertIsInstance(result.caught_exception, exception_group_type)
        self.assertIsInstance(
            result.cancelled_sibling_error, asyncio.CancelledError
        )

        spans_by_name = self._target_spans({
            _EXCEPTION_GROUP_PARENT_SPAN_NAME,
            _FAILING_CHILD_SPAN_NAME,
            _CANCELLED_SIBLING_SPAN_NAME,
        })
        parent = spans_by_name[_EXCEPTION_GROUP_PARENT_SPAN_NAME]
        failing_child_span = spans_by_name[_FAILING_CHILD_SPAN_NAME]
        cancelled_sibling_span = spans_by_name[_CANCELLED_SIBLING_SPAN_NAME]
        parent_context = self._span_context(parent)
        failing_context = self._span_context(failing_child_span)
        cancelled_context = self._span_context(cancelled_sibling_span)
        failing_parent = failing_child_span.parent
        cancelled_parent = cancelled_sibling_span.parent
        cancelled_attributes = cancelled_sibling_span.attributes or {}

        self.assertIsNotNone(failing_parent)
        self.assertIsNotNone(cancelled_parent)
        assert failing_parent is not None
        assert cancelled_parent is not None
        self.assertEqual(failing_parent.span_id, parent_context.span_id)
        self.assertEqual(cancelled_parent.span_id, parent_context.span_id)
        self.assertEqual(failing_context.trace_id, parent_context.trace_id)
        self.assertEqual(cancelled_context.trace_id, parent_context.trace_id)
        self.assertNotEqual(failing_context.span_id, cancelled_context.span_id)
        self.assertEqual(
            result.cancelled_sibling_span_ids, [cancelled_context.span_id]
        )
        self.assertIsNotNone(cancelled_sibling_span.end_time)
        self.assertIn(SpanAttributes.CALL.FUNCTION, cancelled_attributes)
        self.assertNotIn(SpanAttributes.CALL.RETURN, cancelled_attributes)
        self.assertNotIn(SpanAttributes.CALL.ERROR, cancelled_attributes)
        self._assert_context_restored(
            result.context_before,
            result.context_after_exception_group,
        )
        self._assert_context_restored(
            result.context_before,
            result.context_after_new_recording,
        )

    def test_cancelled_finalization_failure_preserves_cancelled_error(
        self,
    ) -> None:
        context_before = self._context_snapshot()
        child_started = asyncio.Event()
        wait_for_cancellation = asyncio.Event()
        child_cancelled_errors: list[asyncio.CancelledError] = []
        propagated_cancelled_errors: list[asyncio.CancelledError] = []

        def raising_attributes(
            _ret: Any,
            _exception: Exception | None,
            *_args: Any,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            raise ValueError("expected finalization failure")

        @instrument(
            name=_CANCELLED_CHILD_SPAN_NAME,
            attributes=raising_attributes,
        )
        async def cancelled_child() -> None:
            child_started.set()
            try:
                await wait_for_cancellation.wait()
            except asyncio.CancelledError as error:
                child_cancelled_errors.append(error)
                raise

        @instrument(name=_CANCEL_PARENT_SPAN_NAME)
        async def cancel_parent() -> None:
            task = asyncio.create_task(cancelled_child())
            await child_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            propagated_cancelled_errors.append(raised.value)

        async def run_scenario() -> None:
            with self._recording_context():
                await cancel_parent()

        with self.assertLogs(
            "trulens.core.otel.instrument", level="ERROR"
        ) as logs:
            asyncio.run(run_scenario())

        spans_by_name = self._target_spans({
            _CANCEL_PARENT_SPAN_NAME,
            _CANCELLED_CHILD_SPAN_NAME,
        })
        parent = spans_by_name[_CANCEL_PARENT_SPAN_NAME]
        child = spans_by_name[_CANCELLED_CHILD_SPAN_NAME]
        parent_context = self._span_context(parent)
        child_context = self._span_context(child)
        child_parent = child.parent

        self.assertEqual(len(child_cancelled_errors), 1)
        self.assertEqual(len(propagated_cancelled_errors), 1)
        self.assertIs(propagated_cancelled_errors[0], child_cancelled_errors[0])
        self.assertIsNotNone(child_parent)
        assert child_parent is not None
        self.assertEqual(child_parent.span_id, parent_context.span_id)
        self.assertEqual(child_context.trace_id, parent_context.trace_id)
        self.assertIsNotNone(child.end_time)
        self.assertTrue(
            any(
                "Error finalizing span during cancellation." in message
                for message in logs.output
            )
        )
        self._assert_context_restored(
            context_before,
            self._context_snapshot(),
        )

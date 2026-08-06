from __future__ import annotations

import datetime
import json
import logging
import threading
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import weakref

import pandas as pd
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.sampling import EvalDecisionReason
from trulens.core.sampling import SamplingController
from trulens.core.sampling import ingest_eval_active
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

if TYPE_CHECKING:
    from trulens.core.app import App

logger = logging.getLogger(__name__)


def _coerce_attribute_dict(value: Any) -> Dict[str, Any]:
    """Coerce a row's attribute column to a dict.

    Some connectors (notably the Snowflake account-event-table path via
    ``Session.sql(...).to_pandas()``) return VARIANT/OBJECT columns as JSON
    strings rather than dicts. Normalize defensively here so consumers can
    always treat the value as a dict.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# When computing feedbacks, we only consider events that ended after a certain
# time so that we don't have to routinely scan all the events. Unfortunately,
# the event table doesn't have any timestamp for when a row was added so we use
# the "TIMESTAMP" column which is when the span/event ended.
# This is still problematic because a span can end and then take a while to be
# ingested into the event table. To get around this, we subtract a time delta
# from the last processed time to allow for some leeway.
_PROCESSED_TIME_DELTA = datetime.timedelta(hours=1)


class Evaluator:
    def __init__(self, app: App):
        self._app_ref = weakref.ref(app)
        self._app_name = app.app_name
        self._app_version = app.app_version
        self._thread = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._compute_feedbacks_lock = threading.Lock()
        self._conversation_compute_lock = threading.Lock()
        self._conversation_jobs_lock = threading.Lock()
        self._conversation_jobs: List[Tuple[str, Tuple[str, ...]]] = []
        self._conversation_sampling_decisions = {}
        self._record_id_to_event_count = pd.Series(dtype=int)
        self._sampled_out_record_ids: set = set()
        """Record IDs skipped by sampling.  Tracked separately from
        ``_record_id_to_event_count`` so that explicit ``compute_now``
        calls can still reach them for backfill."""
        self._processed_time = None
        self._last_error: Optional[BaseException] = None

    def enqueue_conversation(
        self, conversation_id: str, record_ids: List[str]
    ) -> None:
        """Queue one context-bounded conversation evaluation."""
        if not record_ids:
            return
        job = (conversation_id, tuple(record_ids))
        with self._conversation_jobs_lock:
            if job not in self._conversation_jobs:
                self._conversation_jobs.append(job)
        self._wake_event.set()

    def _get_exact_batch_events(
        self, record_ids: Tuple[str, ...]
    ) -> Optional[pd.DataFrame]:
        """Get a complete event batch once every requested root is visible."""
        events = self._app_ref().connector.get_events(
            app_name=self._app_name,
            app_version=self._app_version,
            record_ids=list(record_ids),
            start_time=None,
        )
        if events is None or events.empty:
            return None
        events = events.copy()
        events["record_attributes"] = events["record_attributes"].apply(
            _coerce_attribute_dict
        )
        if "trace" in events.columns:
            events["trace"] = events["trace"].apply(_coerce_attribute_dict)
        visible_root_ids = {
            attributes.get(SpanAttributes.RECORD_ID)
            for attributes in events["record_attributes"]
            if attributes.get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.RECORD_ROOT
        }
        if any(record_id not in visible_root_ids for record_id in record_ids):
            return None
        return events

    def _compute_queued_conversations(self) -> None:
        """Compute queued conversations whose complete record batch is visible."""
        with self._conversation_jobs_lock:
            jobs = list(self._conversation_jobs)

        for job in jobs:
            conversation_id, record_ids = job
            events = self._get_exact_batch_events(record_ids)
            if events is None:
                continue
            controller = TruSession()._sampling_controller
            should_eval = True
            sampling_meta = {}
            reason = EvalDecisionReason.NOT_CONFIGURED.value
            if job in self._conversation_sampling_decisions:
                should_eval, sampling_meta, reason = (
                    self._conversation_sampling_decisions[job]
                )
            elif controller is not None:
                sampling_key = (
                    f"conversation:{conversation_id}:{','.join(record_ids)}"
                )
                should_eval, sampling_meta = controller.should_evaluate(
                    record_id=sampling_key,
                    app_name=self._app_name,
                )
                reason = sampling_meta.get("eval_decision_reason")
                self._conversation_sampling_decisions[job] = (
                    should_eval,
                    sampling_meta,
                    reason,
                )
            if (
                controller is not None
                and reason != EvalDecisionReason.NOT_CONFIGURED.value
            ):
                _emit_sampling_decision_span(
                    record_id=record_ids[-1],
                    app_name=self._app_name,
                    app_version=self._app_version,
                    events=events,
                    sampling_meta=sampling_meta,
                )
            if not should_eval:
                with self._conversation_jobs_lock:
                    if job in self._conversation_jobs:
                        self._conversation_jobs.remove(job)
                self._conversation_sampling_decisions.pop(job, None)
                continue

            token = (
                ingest_eval_active.set(True)
                if controller is not None
                and reason != EvalDecisionReason.NOT_CONFIGURED.value
                else None
            )
            with self._conversation_compute_lock:
                try:
                    self._app_ref().compute_feedbacks(
                        raise_error_on_no_feedbacks_computed=True,
                        events=events,
                        metric_scope="conversation",
                    )
                    TruSession().force_flush()
                except Exception as e:
                    logger.warning(
                        "Error computing conversation feedbacks "
                        "(conversation_id=%s, record_ids=%s): %s\n%s",
                        conversation_id,
                        list(record_ids),
                        e,
                        traceback.format_exc(),
                    )
                    continue
                finally:
                    if token is not None:
                        ingest_eval_active.reset(token)

            with self._conversation_jobs_lock:
                if job in self._conversation_jobs:
                    self._conversation_jobs.remove(job)
            self._conversation_sampling_decisions.pop(job, None)

    def _events_under_record_root(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Get events that are under the record root.

        Args:
            events: A pandas DataFrame of events.

        Returns:
            The events that are under the record root.
        """
        # Construct a tree of events.
        record_roots = []
        span_id_to_children = {
            event["trace"].get("span_id"): []
            for _, event in events.iterrows()
            if isinstance(event.get("trace"), dict)
        }
        for i, event in events.iterrows():
            trace = event.get("trace")
            if not isinstance(trace, dict):
                continue
            # ``GET_AI_OBSERVABILITY_EVENTS`` returns ``TRACE`` objects with
            # only ``span_id``/``trace_id``; the parent linkage lives on
            # ``RECORD.parent_span_id``. Look it up there as a fallback.
            parent_id = trace.get("parent_id")
            if parent_id is None:
                record = event.get("record")
                if isinstance(record, dict):
                    parent_id = record.get("parent_span_id")
            if parent_id and parent_id in span_id_to_children:
                span_id_to_children[parent_id].append(event)
            record_attrs = event.get("record_attributes")
            if (
                isinstance(record_attrs, dict)
                and record_attrs.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.RECORD_ROOT
            ):
                record_roots.append(event)
        # Get events under the record root.
        if len(record_roots) != 1:
            return pd.DataFrame()
        ret = []
        q = [record_roots[0]]
        while q:
            curr_event = q.pop(0)
            ret.append(curr_event)
            span_id = curr_event["trace"].get("span_id")
            q.extend(span_id_to_children.get(span_id, []))
        return pd.DataFrame(ret)

    def _get_record_id_to_unprocessed_events(
        self,
        record_ids: Optional[List[str]],
        start_time: Optional[datetime.datetime],
        force: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get events for the app that weren't yet used for feedback computation.

        Args:
            record_ids:
                Optional list of record IDs to filter events by. If None, all
                unprocessed events will be returned.
            force:
                If True, bypass the internal bookkeeping filters
                (``_record_id_to_event_count`` and
                ``_sampled_out_record_ids``) and return events for the
                requested records regardless of prior processing state.
                Used by ``compute_now`` so that explicitly requested
                backfills always run.

        Returns:
            A dict from record id to a pandas DataFrame of all events from that
            record. Only records that aren't fully processed will be included.
        """
        # When specific record_ids are provided, don't apply start_time filter.
        # We want ALL events for those specific records regardless of when they
        # were created. The start_time filter is only useful when scanning for
        # new records (record_ids=None) to avoid re-processing old events.
        effective_start_time = None if record_ids is not None else start_time
        events = self._app_ref().connector.get_events(
            app_name=self._app_ref().app_name,
            app_version=self._app_ref().app_version,
            record_ids=record_ids,
            start_time=effective_start_time,
        )
        if events is None or len(events) == 0:
            return {}
        # Defensively coerce VARIANT/JSON-string attribute columns to dicts.
        # Some connectors (e.g. the Snowflake account event table path) return
        # these columns as serialized JSON.
        events["record_attributes"] = events["record_attributes"].apply(
            _coerce_attribute_dict
        )
        if "trace" in events.columns:
            events["trace"] = events["trace"].apply(_coerce_attribute_dict)
        record_ids = events["record_attributes"].apply(
            lambda curr: curr.get(SpanAttributes.RECORD_ID)
        )
        record_id_to_events = events.groupby(record_ids)
        record_id_to_events_under_record_root = {
            k: self._events_under_record_root(events.loc[v])
            for k, v in record_id_to_events.groups.items()
        }
        ret = {}
        for (
            record_id,
            events_under_record_root,
        ) in record_id_to_events_under_record_root.items():
            if force:
                # Force path: return events regardless of bookkeeping.
                # Duplicate evaluation is prevented downstream by
                # _remove_already_computed_feedbacks in computer.py.
                ret[record_id] = events_under_record_root
                continue
            count = len(events_under_record_root)
            # On the automatic path, also skip records that were
            # already declined by sampling.
            if record_id in self._sampled_out_record_ids:
                continue
            if (
                record_id not in self._record_id_to_event_count
                or count > self._record_id_to_event_count[record_id]
            ):
                ret[record_id] = events_under_record_root
        return ret

    def _compute_feedbacks(
        self,
        record_ids: Optional[List[str]] = None,
        in_evaluator_thread: bool = True,
        lock: Optional[threading.Lock] = None,
    ) -> None:
        new_processed_time = datetime.datetime.now()
        if lock is None:
            lock = self._compute_feedbacks_lock
        with lock:
            if self._processed_time is None:
                logger.info("Processing all events.")
            else:
                logger.info(
                    f"Processing all events from {self._processed_time}"
                )
            # When explicit record_ids are provided, this is a force
            # path: fetch those records regardless of prior
            # processing/sampling state.  Duplicate evaluation is
            # prevented downstream by _remove_already_computed_feedbacks.
            force = record_ids is not None
            record_id_to_events = self._get_record_id_to_unprocessed_events(
                record_ids,
                self._processed_time,
                force=force,
            )

            # Sampling is only applied on the automatic ingest path
            # (in_evaluator_thread=True).  Explicit compute_now() calls
            # always evaluate everything.
            controller: Optional[SamplingController] = None
            if in_evaluator_thread:
                controller = TruSession()._sampling_controller

            for record_id, events in record_id_to_events.items():
                # --- sampling gate (ingest path only) ---
                if controller is not None:
                    should_eval, sampling_meta = controller.should_evaluate(
                        record_id=record_id,
                        app_name=self._app_name,
                    )
                    reason = sampling_meta.get("eval_decision_reason")

                    # NOT_CONFIGURED means sampling doesn't apply to
                    # this app — fall through to evaluate with no span
                    # overhead.  Only emit decision spans for records
                    # that are actually in scope for sampling.
                    if reason != EvalDecisionReason.NOT_CONFIGURED.value:
                        _emit_sampling_decision_span(
                            record_id=record_id,
                            app_name=self._app_name,
                            app_version=self._app_version,
                            events=events,
                            sampling_meta=sampling_meta,
                        )

                    if not should_eval:
                        logger.debug(
                            "Skipping evaluation for record_id=%s: %s",
                            record_id,
                            reason,
                        )
                        # Track as sampled-out, NOT as processed.
                        # This lets compute_now() still reach these
                        # records for explicit backfill.
                        self._sampled_out_record_ids.add(record_id)
                        continue

                # Set the ingest flag so record_cost() in computer.py
                # only charges the budget on the ingest path.  A batch
                # backfill must not burn the daily budget.
                token = (
                    ingest_eval_active.set(True)
                    if controller is not None
                    else None
                )
                try:
                    self._app_ref().compute_feedbacks(
                        raise_error_on_no_feedbacks_computed=False,
                        events=events,
                        metric_scope="record",
                    )
                except Exception as e:
                    logger.warning(
                        f"Error computing feedbacks in evaluator thread (record_id={record_id}): {e}\n{traceback.format_exc()}"
                    )
                finally:
                    if token is not None:
                        ingest_eval_active.reset(token)
                    self._record_id_to_event_count[record_id] = len(events)
                    TruSession().force_flush()
                if in_evaluator_thread and self._stop_event.is_set():
                    break
        if not record_ids:
            self._processed_time = new_processed_time - _PROCESSED_TIME_DELTA

    def _run_evaluator(self) -> None:
        """Background thread that periodically computes feedback for events.

        Per-iteration errors are logged but the loop continues so the
        evaluator does not silently die on transient failures (e.g. a
        malformed row from the event table). The most recent failure is
        retained on ``self._last_error`` so callers can detect it via
        :meth:`get_last_error`.
        """
        while not self._stop_event.is_set():
            try:
                self._compute_queued_conversations()
                self._compute_feedbacks()
            except Exception as e:
                self._last_error = e
                logger.error(
                    f"Evaluator thread encountered an error: {e}\n{traceback.format_exc()}"
                )
            self._wake_event.wait(timeout=10)
            self._wake_event.clear()

    def get_last_error(self) -> Optional[BaseException]:
        """Return the most recent exception observed by the evaluator
        thread, or ``None`` if no error has occurred."""
        return getattr(self, "_last_error", None)

    def start_evaluator(self) -> None:
        """Start the evaluator for the app."""
        # Validate.
        if not is_otel_tracing_enabled():
            raise ValueError(
                "This method is only supported for OTEL Tracing. Please enable OTEL tracing in the environment!"
            )
        if self._thread is not None:
            raise RuntimeError(
                "Evaluator thread already started. Please stop it before starting a new one."
            )
        # Create and start the evaluator thread.
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_evaluator,
            daemon=True,
            name=f"evaluator_thread(app_name={self._app_name}, app_version={self._app_version})",
        )
        self._thread.start()

    def stop_evaluator(self) -> None:
        """Stop the evaluator for the app.

        This is only supported for OTEL Tracing.
        """
        if self._thread is None:
            logger.warning("No evaluator thread is running.")
            return
        # Signal the thread to stop.
        self._stop_event.set()
        self._wake_event.set()
        # If called from within the evaluator thread, skip join to avoid deadlock.
        if threading.current_thread() is self._thread:
            logger.info(
                f"Stop requested from evaluator thread itself; skipping join for (app_name={self._app_name}, app_version={self._app_version})."
            )
        else:
            # Give the thread a reasonable time to exit gracefully.
            self._thread.join(timeout=300)
            # If thread is still alive after timeout, log a warning.
            if self._thread.is_alive():
                logger.warning(
                    f"Evaluator thread (app_name={self._app_name}, app_version={self._app_version}) did not terminate gracefully within timeout."
                )
            else:
                logger.info(
                    f"Stopped evaluator thread (app_name={self._app_name}, app_version={self._app_version})."
                )
        # Reset for potential future restart.
        self._thread = None
        self._stop_event.clear()

    def compute_now(
        self,
        record_ids: Optional[List[str]],
        lock: Optional[threading.Lock] = None,
    ) -> None:
        """Trigger immediate computation.

        Args:
            record_ids:
                Optional list of record ids to compute feedbacks for. If None,
                computes feedbacks for all unprocessed records.
            lock:
                Optional lock to use for the computation. If None, will use the
                default lock.
        """
        self._compute_feedbacks(
            record_ids, in_evaluator_thread=False, lock=lock
        )

    def compute_conversation_now(self, record_ids: List[str]) -> None:
        """Compute conversation metrics for one exact recording batch."""
        record_ids_tuple = tuple(record_ids)
        with self._conversation_compute_lock:
            events = self._get_exact_batch_events(record_ids_tuple)
            if events is None:
                raise RuntimeError(
                    f"Records are not visible for conversation evaluation: {record_ids}"
                )
            self._app_ref().compute_feedbacks(
                raise_error_on_no_feedbacks_computed=False,
                events=events,
                metric_scope="conversation",
            )
            TruSession().force_flush()
        with self._conversation_jobs_lock:
            matching_jobs = [
                job
                for job in self._conversation_jobs
                if job[1] == record_ids_tuple
            ]
            for job in matching_jobs:
                self._conversation_jobs.remove(job)
                self._conversation_sampling_decisions.pop(job, None)

    def __del__(self):
        try:
            if self._thread is not None:
                logger.info(
                    f"Stopping evaluator thread during garbage collection (app_name={self._app_name}, app_version={self._app_version})."
                )
                self.stop_evaluator()
        except Exception:
            # During interpreter shutdown, some modules might be already
            # unloaded so we can't rely on the logger or other modules being
            # available
            pass


def _emit_sampling_decision_span(
    record_id: str,
    app_name: str,
    app_version: str,
    events: pd.DataFrame,
    sampling_meta: Dict[str, Any],
) -> None:
    """Emit a lightweight span recording the sampling decision for a record.

    One of these spans is created for every record that flows through the
    evaluator (both evaluated and skipped), so that
    ``get_records_and_feedback`` can project a ``sampled`` column and the
    dashboard can show coverage.
    """
    try:
        from opentelemetry import trace as otel_trace
        from trulens.experimental.otel_tracing.core.session import (
            TRULENS_SERVICE_NAME,
        )
        from trulens.experimental.otel_tracing.core.span import (
            set_general_span_attributes,
        )

        tracer = otel_trace.get_tracer_provider().get_tracer(
            TRULENS_SERVICE_NAME
        )

        # Extract app_id from record root event if available.
        app_id = None
        if events is not None and not events.empty:
            for _, event in events.iterrows():
                res_attrs = event.get("resource_attributes")
                if isinstance(res_attrs, str):
                    res_attrs = json.loads(res_attrs)
                if isinstance(res_attrs, dict):
                    app_id = res_attrs.get(ResourceAttributes.APP_ID)
                    if app_id:
                        break

        with tracer.start_as_current_span("eval_decision") as span:
            set_general_span_attributes(
                span, SpanAttributes.SpanType.EVAL_DECISION
            )
            span.set_attribute(SpanAttributes.RECORD_ID, record_id)
            span.set_attribute(
                SpanAttributes.EVAL_DECISION.SAMPLE_RATE,
                sampling_meta.get("sample_rate", 1.0),
            )
            span.set_attribute(
                SpanAttributes.EVAL_DECISION.EVAL_DECISION_REASON,
                sampling_meta.get(
                    "eval_decision_reason",
                    EvalDecisionReason.EVALUATED.value,
                ),
            )
            # Set resource attributes so the span is associated with
            # the correct app.
            span.set_attribute(ResourceAttributes.APP_NAME, app_name)
            span.set_attribute(ResourceAttributes.APP_VERSION, app_version)
            if app_id:
                span.set_attribute(ResourceAttributes.APP_ID, app_id)
    except Exception as e:
        logger.debug(
            "Failed to emit sampling decision span for record_id=%s: %s",
            record_id,
            e,
        )

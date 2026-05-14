from __future__ import annotations

import datetime
import json
import logging
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import weakref

import pandas as pd
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
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
        self._compute_feedbacks_lock = threading.Lock()
        self._record_id_to_event_count = pd.Series(dtype=int)
        self._processed_time = None
        self._last_error: Optional[BaseException] = None

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
    ) -> Dict[str, pd.DataFrame]:
        """
        Get events for the app that weren't yet used for feedback computation.

        Args:
            record_ids:
                Optional list of record IDs to filter events by. If None, all
                unprocessed events will be returned.

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
            count = len(events_under_record_root)
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
            record_id_to_events = self._get_record_id_to_unprocessed_events(
                record_ids,
                self._processed_time,
            )
            for record_id, events in record_id_to_events.items():
                try:
                    self._app_ref().compute_feedbacks(
                        raise_error_on_no_feedbacks_computed=False,
                        events=events,
                    )
                except Exception as e:
                    logger.warning(
                        f"Error computing feedbacks in evaluator thread (record_id={record_id}): {e}\n{traceback.format_exc()}"
                    )
                finally:
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
                self._compute_feedbacks()
            except Exception as e:
                self._last_error = e
                logger.error(
                    f"Evaluator thread encountered an error: {e}\n{traceback.format_exc()}"
                )
            for _ in range(100):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

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

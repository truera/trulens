from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import pandas as pd
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

if TYPE_CHECKING:
    from trulens.core.app import App


class Recording:
    records: List[Record]

    def __init__(self, tru_app: App) -> None:
        self._tru_app = tru_app
        self.records = []

    def add_record_id(self, record_id: str) -> None:
        self.records.append(Record(self._tru_app, record_id))

    def get(self) -> Record:
        """
        Assumes there is exactly one record ID in the recording and returns it.

        Returns:
            The single record ID of the recording.
        """
        if len(self.records) == 0:
            raise RuntimeError("No records found!")
        if len(self.records) != 1:
            raise RuntimeError("There are multiple records!")
        self.records[0]._wait_for_record()
        return self.records[0]

    def __getitem__(self, index: int) -> Record:
        self.records[index]._wait_for_record()
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)

    def retrieve_feedback_results(self, timeout: float = 180) -> pd.DataFrame:
        """Retrieve feedback results for all records in the recording.

        Args:
            timeout: Timeout in seconds to wait.

        Returns:
            A dataframe with records as rows and feedbacks as columns.
        """
        return self._tru_app.retrieve_feedback_results(
            record_ids=[curr.record_id for curr in self.records],
            timeout=timeout,
        )

    # TODO(otel): record_metadata maybe?


# TODO(otel):
# This largely duplicates the functionality of
# `trulens.core.schema.record.Record` which was used for pre-OTel. Eventually,
# we should move to combining the two.
# Other functionality that needs to be added:
# 1. self.feedback_results
# 2. w/e get_feedback_result needs
# 3. w/e trulens_trace needs
# 4. w/e trulens_feedback needs
# 5. w/e display needs
# 6. w/e langchain_quickstart is doing.
# 7. w/e TruSession::run_feedback_functions needs.
class Record:
    """A class to represent a record."""

    def __init__(self, tru_app: App, record_id: str) -> None:
        self._tru_app = tru_app
        self._in_database = False
        self._events = None
        self._record_root_event = None
        self.record_id = record_id

    @property
    def main_input(self) -> Any:
        self._wait_for_record()
        record_root_event = self._get_record_root_event()
        return record_root_event["record_attributes"].get(
            SpanAttributes.RECORD_ROOT.INPUT
        )

    @property
    def main_output(self) -> Any:
        self._wait_for_record()
        record_root_event = self._get_record_root_event()
        return record_root_event["record_attributes"].get(
            SpanAttributes.RECORD_ROOT.OUTPUT
        )

    @property
    def main_error(self) -> Any:
        self._wait_for_record()
        record_root_event = self._get_record_root_event()
        return record_root_event["record_attributes"].get(
            SpanAttributes.RECORD_ROOT.ERROR
        )

    @property
    def meta(self) -> Any:
        # TODO(otel): Currently there's no metadata in OTel records!
        return None

    def _wait_for_record(self) -> None:
        if not self._in_database:
            TruSession().wait_for_records(
                record_ids=[self.record_id], timeout=180
            )
            self._in_database = True

    def _get_events(self, refresh: bool = False) -> pd.DataFrame:
        self._wait_for_record()
        if self._events is not None and not refresh:
            return self._events
        self._events = TruSession().connector.get_events(
            record_ids=[self.record_id]
        )
        if len(self._events) == 0:
            raise RuntimeError(
                f"No events found for record ID {self.record_id}!"
            )
        return self._events

    def _get_record_root_event(self, refresh: bool = False) -> pd.Series:
        self._wait_for_record()
        if self._record_root_event is not None and not refresh:
            return self._record_root_event
        events = self._get_events(refresh=refresh)
        record_root_events = [
            curr
            for _, curr in events.iterrows()
            if curr["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.RECORD_ROOT
        ]
        if len(record_root_events) == 0:
            raise RuntimeError(
                f"No record root event found for record ID {self.record_id}!"
            )
        if len(record_root_events) > 1:
            raise RuntimeError(
                f"Multiple record root events found for record ID {self.record_id}!"
            )
        self._record_root_event = record_root_events[0]
        return self._record_root_event

    def retrieve_feedback_results(self, timeout: float = 180) -> pd.DataFrame:
        """Retrieve feedback results for the record.

        Args:
            timeout: Timeout in seconds to wait.

        Returns:
            A dataframe with a single row and feedbacks as columns.
        """
        return self._tru_app.retrieve_feedback_results(
            record_ids=[self.record_id], timeout=timeout
        )

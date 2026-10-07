"""
Tests for OTEL app.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

from opentelemetry.util.types import AttributeValue
import pandas as pd
import sqlalchemy as sa
from trulens.core.otel.instrument import instrument
from trulens.core.schema.event import EventRecordType
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.test import TruTestCase
from tests.util.df_comparison import (
    compare_dfs_accounting_for_ids_and_timestamps,
)

logger = logging.getLogger(__name__)

ENVIRONMENT_DEPENDENT_RESOURCE_ATTRIBUTES: Tuple[str, ...] = (
    # Version of the installed opentelemetry-sdk.
    "telemetry.sdk.version",
    # Installed trulens-core version. Absent when trulens-core is importable
    # but not installed as a distribution.
    "service.version",
    # Random UUID per process, added by default by newer opentelemetry-sdk
    # releases and absent from older ones.
    "service.instance.id",
)
"""Resource attribute keys removed from both sides of a golden comparison
(and from newly written golden files), because their presence and value depend
on the installed packages and the process rather than on the app under test."""

SPAN_EVENT_TIMESTAMP_PLACEHOLDER = 0
"""Replaces span event timestamps, which are raw nanosecond wall clock values
that the id and timestamp aware comparison cannot line up."""

EXCEPTION_STACKTRACE_PLACEHOLDER = "<exception.stacktrace>"
"""Replaces the `exception.stacktrace` attribute of span events, which holds
absolute file paths, line numbers and a Python version specific layout. The
`exception.type` and `exception.message` attributes are still compared."""


def normalize_environment_dependent_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an events dataframe with the fields that vary across
    environments (but not across runs of the same app) normalized.

    See `ENVIRONMENT_DEPENDENT_RESOURCE_ATTRIBUTES`,
    `SPAN_EVENT_TIMESTAMP_PLACEHOLDER` and `EXCEPTION_STACKTRACE_PLACEHOLDER`.
    """

    def normalize_resource_attributes(attributes: Dict) -> Dict:
        return {
            k: v
            for k, v in attributes.items()
            if k not in ENVIRONMENT_DEPENDENT_RESOURCE_ATTRIBUTES
        }

    def normalize_event(event: Dict) -> Dict:
        event = dict(event)
        if "timestamp" in event:
            event["timestamp"] = SPAN_EVENT_TIMESTAMP_PLACEHOLDER
        attributes = event.get("attributes")
        if isinstance(attributes, dict) and (
            "exception.stacktrace" in attributes
        ):
            event["attributes"] = {
                **attributes,
                "exception.stacktrace": EXCEPTION_STACKTRACE_PLACEHOLDER,
            }
        return event

    def normalize_record(record: Dict) -> Dict:
        if not record.get("events"):
            return record
        return {
            **record,
            "events": [normalize_event(e) for e in record["events"]],
        }

    ret = df.copy()
    ret["resource_attributes"] = ret["resource_attributes"].apply(
        normalize_resource_attributes
    )
    ret["record"] = ret["record"].apply(normalize_record)
    return ret


class OtelTestCase(TruTestCase):
    _orig_TRULENS_OTEL_TRACING: Optional[str] = None
    _db_path: Optional[str] = None

    @classmethod
    def setUpClass(cls) -> None:
        # Each test class gets completely fresh state
        cls._orig_TRULENS_OTEL_TRACING = os.environ.get("TRULENS_OTEL_TRACING")
        os.environ["TRULENS_OTEL_TRACING"] = "1"
        instrument.enable_all_instrumentation()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        instrument.disable_all_instrumentation()
        if cls._orig_TRULENS_OTEL_TRACING is not None:
            os.environ["TRULENS_OTEL_TRACING"] = cls._orig_TRULENS_OTEL_TRACING
        else:
            if "TRULENS_OTEL_TRACING" in os.environ:
                del os.environ["TRULENS_OTEL_TRACING"]
        return super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()  # clears any previous TruSession singleton
        # Use a unique temp file per test to avoid SQLite file
        # contention when pytest-xdist runs multiple workers in
        # parallel (each worker shares the same working directory).
        fd, self._db_path = tempfile.mkstemp(
            suffix=".sqlite", prefix="trulens_test_"
        )
        os.close(fd)
        os.unlink(self._db_path)  # TruSession will create it
        tru_session = TruSession(database_url=f"sqlite:///{self._db_path}")
        tru_session.reset_database()
        # Do NOT clear the singleton here — the test must reuse this
        # TruSession (and its temp DB) when it calls TruSession().

    def tearDown(self) -> None:
        tru_session = TruSession()
        tru_session.force_flush()
        super().tearDown()  # clears the singleton
        # Clean up the temp database file.
        if self._db_path and os.path.exists(self._db_path):
            try:
                os.unlink(self._db_path)
            except OSError:
                logger.debug(
                    "Failed to remove temp OTEL test database: %s",
                    self._db_path,
                    exc_info=True,
                )

    @staticmethod
    def _get_events() -> pd.DataFrame:
        tru_session = TruSession()
        tru_session.force_flush()
        db = tru_session.connector.db
        with db.session.begin() as db_session:
            q = sa.select(db.orm.Event).order_by(db.orm.Event.start_timestamp)
            return pd.read_sql(q, db_session.bind)

    @staticmethod
    def _convert_column_types(df: pd.DataFrame) -> None:
        # Writing to CSV and the reading back causes some type issues so we
        # hackily convert things here.
        df["event_id"] = df["event_id"].apply(str)
        df["record_type"] = df["record_type"].apply(
            lambda x: (
                EventRecordType(x[len("EventRecordType.") :])
                if x.startswith("EventRecordType.")
                else EventRecordType(x)
            )
        )
        df["start_timestamp"] = df["start_timestamp"].apply(pd.Timestamp)
        df["timestamp"] = df["timestamp"].apply(pd.Timestamp)
        for json_column in [
            "record",
            "record_attributes",
            "resource_attributes",
            "trace",
        ]:
            df[json_column] = df[json_column].apply(lambda x: eval(x))

    def _compare_events_to_golden_dataframe(
        self,
        golden_filename: str,
        ignore_locators: Optional[List[str]] = None,
        regex_replacements: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        tru_session = TruSession()
        tru_session.force_flush()
        actual = normalize_environment_dependent_fields(self._get_events())
        self.write_golden(golden_filename, actual)
        expected = self.load_golden(golden_filename)
        self._convert_column_types(expected)
        expected = normalize_environment_dependent_fields(expected)
        compare_dfs_accounting_for_ids_and_timestamps(
            self,
            expected,
            actual,
            ignore_locators=ignore_locators,
            timestamp_tol=pd.Timedelta("0.02s"),
            regex_replacements=regex_replacements,
        )

    def _compare_record_attributes_to_golden_dataframe(
        self,
        golden_filename: str,
        keys_to_check: List[str] = [SpanAttributes.SPAN_TYPE],
    ) -> None:
        TruSession().force_flush()
        expected = self.load_golden(golden_filename)
        self._convert_column_types(expected)
        actual = self._get_events()
        self.assertEqual(expected.shape, actual.shape)
        for i in range(len(actual)):
            self.assertEqual(
                expected.iloc[i]["record"]["name"],
                actual.iloc[i]["record"]["name"],
            )
            for key in keys_to_check:
                self.assertEqual(
                    expected.iloc[i]["record_attributes"][key],
                    actual.iloc[i]["record_attributes"][key],
                    f"Record attributes do not match for key: {key}",
                )

    def _check_costs(
        self,
        record_attributes: Dict[str, AttributeValue],
        cost_model: str,
        cost_currency: str,
        free: bool,
    ):
        self.assertEqual(
            record_attributes[SpanAttributes.COST.MODEL],
            cost_model,
        )
        self.assertEqual(
            record_attributes[SpanAttributes.COST.CURRENCY],
            cost_currency,
        )
        if free:
            self.assertEqual(
                record_attributes[SpanAttributes.COST.COST],
                0,
            )
        else:
            self.assertGreater(
                record_attributes[SpanAttributes.COST.COST],
                0,
            )
        self.assertGreater(
            record_attributes[SpanAttributes.COST.NUM_TOKENS],
            0,
        )
        self.assertGreater(
            record_attributes[SpanAttributes.COST.NUM_PROMPT_TOKENS],
            0,
        )
        self.assertGreater(
            record_attributes[SpanAttributes.COST.NUM_COMPLETION_TOKENS],
            0,
        )

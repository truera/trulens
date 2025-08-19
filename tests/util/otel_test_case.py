"""
Tests for OTEL app.
"""

import os
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


class OtelTestCase(TruTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["TRULENS_OTEL_TRACING"] = "1"
        instrument.enable_all_instrumentation()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        instrument.disable_all_instrumentation()
        del os.environ["TRULENS_OTEL_TRACING"]
        return super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        tru_session = TruSession()
        tru_session.force_flush()
        tru_session.reset_database()
        self.clear_TruSession_singleton()

    def tearDown(self) -> None:
        tru_session = TruSession()
        tru_session.force_flush()
        return super().tearDown()

    @staticmethod
    def _get_events() -> pd.DataFrame:
        tru_session = TruSession()
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
            lambda x: EventRecordType(x[len("EventRecordType.") :])
            if x.startswith("EventRecordType.")
            else EventRecordType(x)
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
        actual = self._get_events()
        self.write_golden(golden_filename, actual)
        expected = self.load_golden(golden_filename)
        self._convert_column_types(expected)
        if ignore_locators is None:
            ignore_locators = []
        ignore_locators += ["[resource_attributes][telemetry.sdk.version]"]
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

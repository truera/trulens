"""
Tests for OTEL app.
"""

import os

import pandas as pd
import sqlalchemy as sa
from trulens.core.experimental import Feature
from trulens.core.schema.event import EventRecordType
from trulens.core.session import TruSession

from tests.test import TruTestCase
from tests.util.df_comparison import (
    compare_dfs_accounting_for_ids_and_timestamps,
)


class OtelAppTestCase(TruTestCase):
    @classmethod
    def clear_TruSession_singleton(cls) -> None:
        # [HACK!] Clean up any instances of `TruSession` so tests don't
        # interfere with each other.
        for key in [
            curr
            for curr in TruSession._singleton_instances
            if curr[0] == "trulens.core.session.TruSession"
        ]:
            del TruSession._singleton_instances[key]

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["TRULENS_OTEL_TRACING"] = "1"
        cls.clear_TruSession_singleton()
        tru_session = TruSession(
            experimental_feature_flags=[Feature.OTEL_TRACING]
        )
        tru_session.experimental_enable_feature("otel_tracing")
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.clear_TruSession_singleton()
        return super().tearDownClass()

    def setUp(self) -> None:
        tru_session = TruSession(
            experimental_feature_flags=[Feature.OTEL_TRACING]
        )
        tru_session.experimental_enable_feature("otel_tracing")
        tru_session.reset_database()
        return super().setUp()

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

    def _compare_events_to_golden_dataframe(self, golden_filename: str) -> None:
        tru_session = TruSession()
        tru_session.experimental_force_flush()
        actual = self._get_events()
        self.write_golden(golden_filename, actual)
        expected = self.load_golden(golden_filename)
        self._convert_column_types(expected)
        compare_dfs_accounting_for_ids_and_timestamps(
            self,
            expected,
            actual,
            ignore_locators=[
                f"df.iloc[{i}][resource_attributes][telemetry.sdk.version]"
                for i in range(len(expected))
            ],
            timestamp_tol=pd.Timedelta("0.02s"),
        )

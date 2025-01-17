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

"""
Tests for OTEL instrument decorator.
"""

import os
from unittest import main

import pandas as pd
import sqlalchemy as sa
from trulens.apps.custom import TruCustomApp
from trulens.core.schema.event import EventRecordType
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.init import init
from trulens.experimental.otel_tracing.core.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from tests.test import TruTestCase
from tests.util.df_comparison import (
    compare_dfs_accounting_for_ids_and_timestamps,
)


class _TestApp:
    @instrument(span_type=SpanAttributes.SpanType.MAIN)
    def respond_to_query(self, query: str) -> str:
        return f"answer: {self.nested(query)}"

    @instrument(attributes={"nested_attr1": "value1"})
    def nested(self, query: str) -> str:
        return f"nested: {self.nested2(query)}"

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            "nested2_ret": ret,
            "nested2_args[1]": args[1],
        }
    )
    def nested2(self, query: str) -> str:
        nested_result = ""

        try:
            nested_result = self.nested3(query)
        except Exception:
            pass

        return f"nested2: {nested_result}"

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            "nested3_ex": exception.args if exception else None,
            "nested3_ret": ret,
            "selector_name": "special",
            "cows": "moo",
        }
    )
    def nested3(self, query: str) -> str:
        if query == "throw":
            raise ValueError("nested3 exception")
        return "nested3"


class TestOtelInstrument(TruTestCase):
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
        tru_session = TruSession()
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

    def test_instrument_decorator(self) -> None:
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        init(tru_session, debug=True)
        # Create and run app.
        test_app = _TestApp()
        custom_app = TruCustomApp(test_app)
        with custom_app:
            test_app.respond_to_query("test")
        with custom_app:
            test_app.respond_to_query("throw")
        # Compare results to expected.
        GOLDEN_FILENAME = "tests/unit/static/golden/test_otel_instrument__test_instrument_decorator.csv"
        actual = self._get_events()
        self.assertEqual(len(actual), 10)
        self.write_golden(GOLDEN_FILENAME, actual)
        expected = self.load_golden(GOLDEN_FILENAME)
        self._convert_column_types(expected)
        compare_dfs_accounting_for_ids_and_timestamps(
            self,
            expected,
            actual,
            ignore_locators=[
                f"df.iloc[{i}][resource_attributes][telemetry.sdk.version]"
                for i in range(10)
            ],
        )


if __name__ == "__main__":
    main()

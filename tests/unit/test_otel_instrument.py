"""
Tests for OTEL instrument decorator.
"""

from typing import Any, Dict
from unittest import main

import pandas as pd
import sqlalchemy as sa
from trulens.apps.custom import TruCustomApp
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.init import init
from trulens.experimental.otel_tracing.core.instrument import instrument

from tests.test import TruTestCase


class _TestApp:
    @instrument()
    def respond_to_query(self, query: str) -> str:
        return f"answer: {self.nested(query)}"

    @instrument(attributes={"nested_attr1": "value1"})
    def nested(self, query: str) -> str:
        return f"nested: {self.nested2(query)}"

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            "nested2_ret": ret,
            "nested2_args[0]": args[0],
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
    def setUp(self):
        pass

    @staticmethod
    def _get_events() -> pd.DataFrame:
        tru_session = TruSession()
        db = tru_session.connector.db
        with db.session.begin() as db_session:
            q = sa.select(db.orm.Event).order_by(db.orm.Event.start_timestamp)
            return pd.read_sql(q, db_session.bind)

    @staticmethod
    def _convert_column_types(df: pd.DataFrame):
        df["event_id"] = df["event_id"].apply(str)
        df["record_type"] = df["record_type"].apply(lambda x: eval(x))
        df["start_timestamp"] = df["start_timestamp"].apply(pd.Timestamp)
        df["timestamp"] = df["timestamp"].apply(pd.Timestamp)
        for json_column in [
            "record",
            "record_attributes",
            "resource_attributes",
            "trace",
        ]:
            df[json_column] = df[json_column].apply(lambda x: eval(x))

    def _compare_dfs_accounting_for_ids_and_timestamps(
        self, expected: pd.DataFrame, actual: pd.DataFrame
    ):
        """
        Compare two Dataframes are equal, accounting for ids and timestamps.
        That is:
        1. The ids between the two Dataframes may be different, but they have
           to be consistent. That is, if one Dataframe reuses an id in two
           places, then the other must as well.
        2. The timestamps between the two Dataframes may be different, but
           they have to be in the same order.

        Args:
            expected: expected results
            actual: actual results
        """
        id_mapping: Dict[str, str] = {}
        timestamp_mapping: Dict[pd.Timestamp, pd.Timestamp] = {}
        self.assertEqual(len(expected), len(actual))
        self.assertListEqual(list(expected.columns), list(actual.columns))
        for i in range(len(expected)):
            for col in expected.columns:
                self._compare_entity(
                    expected.iloc[i][col],
                    actual.iloc[i][col],
                    id_mapping,
                    timestamp_mapping,
                    is_id=col.endswith("_id"),
                    locator=f"df.iloc[{i}][{col}]",
                )
        # Ensure that the id mapping is a bijection.
        self.assertEqual(
            len(set(id_mapping.values())),
            len(id_mapping),
            "Ids are not a bijection!",
        )
        # Ensure that the timestamp mapping is monotonic.
        prev_value = None
        for curr in sorted(timestamp_mapping.keys()):
            if prev_value is not None:
                self.assertLess(
                    prev_value,
                    timestamp_mapping[curr],
                    "Timestamps are not in the same order!",
                )
            prev_value = timestamp_mapping[curr]

    def _compare_entity(
        self,
        expected: Any,
        actual: Any,
        id_mapping: Dict[str, str],
        timestamp_mapping: Dict[pd.Timestamp, pd.Timestamp],
        is_id: bool,
        locator: str,
    ):
        self.assertEqual(
            type(expected), type(actual), f"Types of {locator} do not match!"
        )
        if is_id:
            self.assertEqual(
                type(expected), str, f"Type of id {locator} is not a string!"
            )
            if expected not in id_mapping:
                id_mapping[expected] = actual
            self.assertEqual(
                id_mapping[expected],
                actual,
                f"Ids of {locator} are not consistent!",
            )
        elif isinstance(expected, dict):
            self.assertEqual(
                expected.keys(),
                actual.keys(),
                f"Keys of {locator} do not match!",
            )
            for k in expected.keys():
                self._compare_entity(
                    expected[k],
                    actual[k],
                    id_mapping,
                    timestamp_mapping,
                    is_id=k.endswith("_id"),
                    locator=f"{locator}[k]",
                )
        elif isinstance(expected, pd.Timestamp):
            if expected not in timestamp_mapping:
                timestamp_mapping[expected] = actual
            self.assertEqual(
                timestamp_mapping[expected],
                actual,
                f"Timestamps of {locator} are not consistent!",
            )
        else:
            self.assertEqual(expected, actual, f"{locator} does not match!")

    def test_deterministic_app_id(self):
        # Set up.
        tru_session = TruSession()
        tru_session.experimental_enable_feature("otel_tracing")
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
        GOLDEN_FILENAME = "tests/unit/static/golden/test_otel_instrument__test_deterministic_app_id.csv"
        actual = self._get_events()
        self.assertEqual(len(actual), 8)
        self.write_golden(GOLDEN_FILENAME, actual)
        expected = self.load_golden(GOLDEN_FILENAME)
        self._convert_column_types(expected)
        self._compare_dfs_accounting_for_ids_and_timestamps(expected, actual)


if __name__ == "__main__":
    main()

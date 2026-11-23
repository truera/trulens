"""Tests for span ingestion SQL in the Snowflake OTEL exporter."""

import unittest
from unittest import mock

import pytest

try:
    from trulens.connectors.snowflake import otel_exporter
except ImportError:
    otel_exporter = None


@pytest.mark.optional
@unittest.skipIf(
    otel_exporter is None, "trulens-connectors-snowflake not installed"
)
class TestIngestSpansFromStage(unittest.TestCase):
    def _ingest(self, input_records_count):
        session = mock.MagicMock()
        session.get_current_database.return_value = "DB"
        session.get_current_schema.return_value = "SCH"
        otel_exporter.TruLensSnowflakeSpanExporter._ingest_spans_from_stage(
            session,
            "file.pb",
            "my_app",
            "v1",
            "run1",
            input_records_count,
            False,
        )
        call = session.sql.call_args
        return call.args[0], call.kwargs["params"]

    def test_none_count_is_not_bound(self):
        # Inside a stored procedure Snowpark binds `None` as the string
        # "None", which the ingestion procedure rejects.
        sql, params = self._ingest(None)
        self.assertIn("'input_record_count', NULL", sql)
        self.assertNotIn(None, params)
        self.assertEqual(sql.count("?"), len(params))

    def test_count_is_bound(self):
        sql, params = self._ingest(3)
        self.assertIn("'input_record_count', ?", sql)
        self.assertEqual(params[-1], 3)
        self.assertEqual(sql.count("?"), len(params))


if __name__ == "__main__":
    unittest.main()

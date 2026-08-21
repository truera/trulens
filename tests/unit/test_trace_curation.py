"""Tests for curating recorded traces into persisted datasets.

Covers the contract from truera/trulens#2702: mapping validation, JSON/text
normalization, the accepted input shapes, callback precedence and failure,
context normalization and metadata selection, idempotency, both error modes,
and batch boundaries.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
from trulens.core import dataset as core_dataset
from trulens.core import session as core_session
from trulens.core.database import sqlalchemy as db_sqlalchemy
from trulens.core.database.connector import default as default_connector
from trulens.core.schema import dataset as dataset_schema

RECORDS = pd.DataFrame([
    {
        "record_id": "record-1",
        "app_name": "support-bot",
        "app_version": "v1",
        "input": "what is trulens?",
        "output": "no idea",
        "corrected_output": "an evaluation library",
        "Groundedness": 0.2,
    },
    {
        "record_id": "record-2",
        "app_name": "support-bot",
        "app_version": "v1",
        "input": "how do i install it?",
        "output": "unclear",
        "corrected_output": "pip install trulens",
        "Groundedness": 0.4,
    },
])

MAPPING = core_dataset.TraceDatasetMapping(
    query="input",
    query_id="record_id",
    expected_response="corrected_output",
    metadata={"groundedness": "Groundedness"},
)


def _clear_tru_session_singletons():
    """Drop any live `TruSession` so each test gets its own database."""

    for key in [
        curr
        for curr in core_session.TruSession._singleton_instances
        if curr[0] == "trulens.core.session.TruSession"
    ]:
        del core_session.TruSession._singleton_instances[key]


class CurationTestCase(unittest.TestCase):
    """Base case giving each test its own file-backed SQLite database."""

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)

        db_path = os.path.join(self._tempdir.name, "trulens.sqlite")
        self.db = db_sqlalchemy.SQLAlchemyDB.from_db_url(f"sqlite:///{db_path}")
        self.db.migrate_database()

        _clear_tru_session_singletons()
        self.addCleanup(_clear_tru_session_singletons)
        self.session = core_session.TruSession(
            connector=default_connector.DefaultDBConnector(database=self.db)
        )

    def curate(self, **kwargs) -> core_dataset.CurationResult:
        """Curate `RECORDS` under the default mapping, overriding as needed."""

        kwargs.setdefault("dataset_name", "production-failures")
        kwargs.setdefault("records", RECORDS)
        kwargs.setdefault("mapping", MAPPING)
        return self.session.curate_records_to_dataset(**kwargs)

    def ground_truths(self, dataset_name="production-failures"):
        return self.session.get_ground_truth(dataset_name=dataset_name)


class TestMappingValidation(CurationTestCase):
    def test_defaults_read_input_and_record_id(self):
        mapping = core_dataset.TraceDatasetMapping()
        self.assertEqual(mapping.query, "input")
        self.assertEqual(mapping.query_id, "record_id")
        self.assertIsNone(mapping.expected_response)
        self.assertEqual(mapping.metadata, {})

    def test_mapped_columns_are_deduplicated_in_order(self):
        mapping = core_dataset.TraceDatasetMapping(
            query="input",
            query_id="input",
            metadata={"a": "input", "b": "Groundedness"},
        )
        self.assertEqual(mapping.mapped_columns(), ["input", "Groundedness"])

    def test_blank_query_column_is_rejected(self):
        with self.assertRaises(ValueError):
            core_dataset.TraceDatasetMapping(query="   ")

    def test_missing_columns_are_reported_before_any_write(self):
        with self.assertRaises(ValueError) as caught:
            self.curate(
                records=RECORDS.drop(columns=["record_id"]),
                mapping=core_dataset.TraceDatasetMapping(
                    query="not_a_column", query_id=None
                ),
            )
        self.assertIn("not_a_column", str(caught.exception))

        # Nothing was written: the dataset has no ground truths.
        self.assertIsNone(self.ground_truths())

    def test_missing_metadata_column_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            self.curate(
                mapping=core_dataset.TraceDatasetMapping(
                    query="input", metadata={"score": "NotAMetric"}
                )
            )
        self.assertIn("NotAMetric", str(caught.exception))

    def test_invalid_on_error_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.curate(on_error="ignore")

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            self.curate(batch_size=0)

    def test_non_dataframe_records_are_rejected(self):
        with self.assertRaises(ValueError):
            self.curate(records=[{"input": "q"}])


class TestNormalization(unittest.TestCase):
    def test_text_passthrough_and_stripping(self):
        self.assertEqual(core_dataset._normalize_text("  hi\n"), "hi")

    def test_missing_values_become_none(self):
        for value in (None, float("nan"), "", "   "):
            self.assertIsNone(core_dataset._normalize_text(value))

    def test_dicts_serialize_with_sorted_keys(self):
        self.assertEqual(
            core_dataset._normalize_text({"b": 1, "a": 2}),
            '{"a": 2, "b": 1}',
        )

    def test_json_strings_are_renormalized(self):
        # The legacy schema hands back parsed values while OTEL hands back
        # strings; both must reduce to the same text.
        self.assertEqual(
            core_dataset._normalize_text('{"b": 1, "a": 2}'),
            core_dataset._normalize_text({"a": 2, "b": 1}),
        )

    def test_malformed_json_is_left_alone(self):
        self.assertEqual(core_dataset._normalize_text("{not json"), "{not json")

    def test_scalar_json_literals_are_left_alone(self):
        self.assertEqual(core_dataset._normalize_text("123"), "123")

    def test_non_string_scalars_become_text(self):
        self.assertEqual(core_dataset._normalize_text(7), "7")

    def test_contexts_from_list_of_dicts(self):
        self.assertEqual(
            core_dataset._normalize_contexts([{"text": "a"}, {"text": "b"}]),
            [{"text": "a"}, {"text": "b"}],
        )

    def test_contexts_from_list_of_strings(self):
        self.assertEqual(
            core_dataset._normalize_contexts(["a", "b"]),
            [{"text": "a"}, {"text": "b"}],
        )

    def test_contexts_from_plain_string(self):
        self.assertEqual(
            core_dataset._normalize_contexts("a chunk"),
            [{"text": "a chunk"}],
        )

    def test_contexts_from_json_string(self):
        self.assertEqual(
            core_dataset._normalize_contexts('["a", "b"]'),
            [{"text": "a"}, {"text": "b"}],
        )

    def test_contexts_from_single_dict(self):
        self.assertEqual(
            core_dataset._normalize_contexts({"text": "a"}), [{"text": "a"}]
        )

    def test_contexts_drop_missing_elements(self):
        self.assertEqual(
            core_dataset._normalize_contexts(["a", None, float("nan")]),
            [{"text": "a"}],
        )

    def test_contexts_missing_becomes_none(self):
        for value in (None, float("nan"), "", []):
            self.assertIsNone(core_dataset._normalize_contexts(value))

    def test_json_safe_coerces_numpy_scalars(self):
        value = core_dataset._json_safe(pd.Series([1.5]).iloc[0])
        self.assertIsInstance(value, float)
        self.assertEqual(value, 1.5)
        # The result must survive a json round trip.
        self.assertEqual(json.loads(json.dumps({"v": value}))["v"], 1.5)

    def test_json_safe_maps_nan_to_none(self):
        self.assertIsNone(core_dataset._json_safe(float("nan")))


class TestCuration(CurationTestCase):
    def test_every_accepted_row_becomes_a_ground_truth(self):
        result = self.curate()

        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.duplicates, 0)
        self.assertEqual(result.rejected, 0)
        self.assertEqual(result.processed, 2)
        self.assertEqual(len(result.ground_truth_ids), 2)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.dataset_name, "production-failures")

        df = self.ground_truths()
        self.assertEqual(len(df), 2)
        self.assertEqual(
            sorted(df["query"]),
            ["how do i install it?", "what is trulens?"],
        )
        self.assertEqual(
            sorted(df["expected_response"]),
            ["an evaluation library", "pip install trulens"],
        )
        self.assertEqual(sorted(df["query_id"]), ["record-1", "record-2"])

    def test_corrected_output_round_trips(self):
        self.curate()
        df = self.ground_truths()
        row = df[df["query_id"] == "record-1"].iloc[0]
        self.assertEqual(row["expected_response"], "an evaluation library")

    def test_provenance_is_preserved_in_metadata(self):
        self.curate()
        df = self.ground_truths()
        meta = df[df["query_id"] == "record-1"].iloc[0]["meta"]

        self.assertEqual(meta["source_record_id"], "record-1")
        self.assertEqual(meta["source_app_name"], "support-bot")
        self.assertEqual(meta["source_app_version"], "v1")
        self.assertEqual(meta["groundedness"], 0.2)

    def test_explicit_metadata_overrides_provenance_default(self):
        self.curate(
            mapping=core_dataset.TraceDatasetMapping(
                query="input",
                query_id="record_id",
                metadata={"source_app_name": "app_version"},
            )
        )
        meta = self.ground_truths().iloc[0]["meta"]
        self.assertEqual(meta["source_app_name"], "v1")

    def test_provenance_skipped_when_columns_absent(self):
        records = RECORDS.drop(columns=["app_name", "app_version"])
        self.curate(
            records=records,
            mapping=core_dataset.TraceDatasetMapping(
                query="input", query_id="record_id"
            ),
        )
        meta = self.ground_truths().iloc[0]["meta"]
        self.assertEqual(meta, {"source_record_id": "record-1"})

    def test_dataset_metadata_is_stored(self):
        result = self.curate(dataset_metadata={"owner": "eval-team"})

        # A dataset id hashes the name and the metadata, so matching the id
        # proves the metadata reached the stored dataset.
        expected = dataset_schema.Dataset(
            name="production-failures", meta={"owner": "eval-team"}
        )
        self.assertEqual(result.dataset_id, expected.dataset_id)
        self.assertEqual(
            set(self.ground_truths()["dataset_id"]), {expected.dataset_id}
        )

    def test_json_inputs_are_resolved_to_text(self):
        # The legacy schema hands back a parsed value while OTEL hands back a
        # string; both spellings must collapse onto one ground truth.
        records = pd.DataFrame([
            {"input": {"question": "what is trulens?"}},
            {"input": '{"question": "what is trulens?"}'},
        ])
        result = self.curate(
            records=records,
            mapping=core_dataset.TraceDatasetMapping(
                query="input", query_id=None
            ),
        )
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.duplicates, 1)

    def test_provenance_keeps_identical_content_distinct(self):
        # A ground truth id covers its metadata, so two records with the same
        # question stay distinct while their provenance differs.
        records = pd.DataFrame([
            {"record_id": "r1", "input": "same question"},
            {"record_id": "r2", "input": "same question"},
        ])
        mapping = core_dataset.TraceDatasetMapping(query="input", query_id=None)

        result = self.curate(records=records, mapping=mapping)
        self.assertEqual(result.accepted, 2)

    def test_provenance_can_be_turned_off_to_deduplicate_on_content(self):
        records = pd.DataFrame([
            {"record_id": "r1", "input": "same question"},
            {"record_id": "r2", "input": "same question"},
        ])
        mapping = core_dataset.TraceDatasetMapping(query="input", query_id=None)

        result = self.curate(
            records=records, mapping=mapping, include_provenance=False
        )

        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.duplicates, 1)
        self.assertEqual(self.ground_truths().iloc[0]["meta"], {})


class TestExpectedContexts(CurationTestCase):
    def test_contexts_round_trip(self):
        records = RECORDS.assign(
            contexts=[["chunk a", "chunk b"], [{"text": "chunk c"}]]
        )
        self.curate(
            records=records,
            mapping=core_dataset.TraceDatasetMapping(
                query="input",
                query_id="record_id",
                expected_chunks="contexts",
            ),
        )

        df = self.ground_truths()
        first = df[df["query_id"] == "record-1"].iloc[0]
        second = df[df["query_id"] == "record-2"].iloc[0]

        self.assertEqual(
            first["expected_chunks"], [{"text": "chunk a"}, {"text": "chunk b"}]
        )
        self.assertEqual(second["expected_chunks"], [{"text": "chunk c"}])


class TestExpectedResponseCallback(CurationTestCase):
    def test_callback_fills_in_missing_corrections(self):
        records = RECORDS.assign(corrected_output=[None, "pip install trulens"])

        self.curate(
            records=records,
            expected_response_fn=lambda row: f"generated for {row['record_id']}",
        )

        df = self.ground_truths()
        self.assertEqual(
            df[df["query_id"] == "record-1"].iloc[0]["expected_response"],
            "generated for record-1",
        )

    def test_mapped_correction_takes_precedence_over_callback(self):
        callback = mock.MagicMock(return_value="from callback")

        self.curate(expected_response_fn=callback)

        df = self.ground_truths()
        self.assertEqual(
            sorted(df["expected_response"]),
            ["an evaluation library", "pip install trulens"],
        )
        callback.assert_not_called()

    def test_callback_is_not_called_without_a_mapped_column(self):
        # With no expected_response mapping at all, the callback supplies
        # every value.
        self.curate(
            mapping=core_dataset.TraceDatasetMapping(
                query="input", query_id="record_id"
            ),
            expected_response_fn=lambda row: "always",
        )
        df = self.ground_truths()
        self.assertEqual(list(df["expected_response"]), ["always", "always"])

    def test_callback_failure_fails_fast(self):
        def boom(row):
            raise RuntimeError("no correction available")

        records = RECORDS.assign(corrected_output=[None, None])

        with self.assertRaises(core_dataset.CurationRowError) as caught:
            self.curate(records=records, expected_response_fn=boom)
        self.assertEqual(caught.exception.reason, "expected_response_fn_failed")
        self.assertIn("no correction available", caught.exception.message)

    def test_callback_failure_can_be_collected(self):
        def boom(row):
            raise RuntimeError("no correction available")

        records = RECORDS.assign(corrected_output=[None, "pip install trulens"])

        result = self.curate(
            records=records, expected_response_fn=boom, on_error="collect"
        )

        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.rejected, 1)
        self.assertEqual(result.errors[0].reason, "expected_response_fn_failed")
        self.assertEqual(result.errors[0].query_id, "record-1")
        self.assertEqual(len(self.ground_truths()), 1)


class TestErrorModes(CurationTestCase):
    def bad_records(self):
        return RECORDS.assign(input=["what is trulens?", None])

    def test_fail_fast_is_the_default(self):
        with self.assertRaises(core_dataset.CurationRowError) as caught:
            self.curate(records=self.bad_records())
        self.assertEqual(caught.exception.reason, "empty_query")

    def test_collect_does_not_publish_malformed_rows(self):
        result = self.curate(records=self.bad_records(), on_error="collect")

        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.rejected, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].reason, "empty_query")
        self.assertEqual(result.errors[0].query_id, "record-2")

        df = self.ground_truths()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["query"], "what is trulens?")

    def test_errors_df_is_inspectable(self):
        result = self.curate(records=self.bad_records(), on_error="collect")
        errors = result.errors_df()

        self.assertEqual(len(errors), 1)
        self.assertEqual(
            list(errors.columns),
            [
                "row_index",
                "query_id",
                "reason",
                "message",
            ],
        )
        self.assertEqual(errors.iloc[0]["reason"], "empty_query")

    def test_whitespace_only_query_is_rejected(self):
        records = RECORDS.assign(input=["what is trulens?", "   "])
        result = self.curate(records=records, on_error="collect")
        self.assertEqual(result.rejected, 1)


class TestIdempotency(CurationTestCase):
    def test_curating_identical_rows_twice_adds_nothing(self):
        first = self.curate()
        second = self.curate()

        self.assertEqual(first.ground_truth_ids, second.ground_truth_ids)
        self.assertEqual(len(self.ground_truths()), 2)

    def test_duplicate_rows_within_one_call_are_counted(self):
        records = pd.concat([RECORDS, RECORDS], ignore_index=True)

        result = self.curate(records=records)

        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.duplicates, 2)
        self.assertEqual(result.processed, 4)
        self.assertEqual(len(self.ground_truths()), 2)

    def test_changing_a_correction_adds_a_new_ground_truth(self):
        self.curate()
        self.curate(
            records=RECORDS.assign(
                corrected_output=["a better answer", "pip install trulens"]
            )
        )
        # Ids are content-addressed, so the edited example is a new row rather
        # than an in-place edit of the old one.
        self.assertEqual(len(self.ground_truths()), 3)


class _RecordingDB:
    """Minimal db stand-in that records the batches it is handed.

    Batch boundaries are a property of the curation loop, not of SQL, so they
    are checked against a stub rather than through a real database.
    """

    def __init__(self):
        self.batch_sizes = []
        self.inserted = []

    def insert_dataset(self, dataset):
        return dataset.dataset_id

    def batch_insert_ground_truth(self, ground_truths):
        self.batch_sizes.append(len(ground_truths))
        self.inserted.extend(ground_truths)
        return [gt.ground_truth_id for gt in ground_truths]


class TestBatching(unittest.TestCase):
    def make_records(self, count: int) -> pd.DataFrame:
        return pd.DataFrame([
            {"record_id": f"r{i}", "input": f"question {i}"}
            for i in range(count)
        ])

    def curate_many(self, count: int, batch_size: int, **kwargs):
        db = _RecordingDB()
        result = core_dataset.curate_records_to_dataset(
            dataset_name="batched",
            records=self.make_records(count),
            db=db,
            mapping=core_dataset.TraceDatasetMapping(
                query="input", query_id="record_id"
            ),
            batch_size=batch_size,
            **kwargs,
        )
        return result, db

    def test_exact_batch_boundary(self):
        result, db = self.curate_many(count=10, batch_size=5)
        self.assertEqual(result.accepted, 10)
        self.assertEqual(db.batch_sizes, [5, 5])

    def test_partial_final_batch(self):
        result, db = self.curate_many(count=7, batch_size=3)
        self.assertEqual(result.accepted, 7)
        self.assertEqual(db.batch_sizes, [3, 3, 1])

    def test_single_batch_when_smaller_than_batch_size(self):
        _, db = self.curate_many(count=2, batch_size=100)
        self.assertEqual(db.batch_sizes, [2])

    def test_batch_size_of_one(self):
        _, db = self.curate_many(count=3, batch_size=1)
        self.assertEqual(db.batch_sizes, [1, 1, 1])

    def test_every_row_is_written_exactly_once(self):
        _, db = self.curate_many(count=25, batch_size=4)
        self.assertEqual(sum(db.batch_sizes), 25)
        self.assertEqual(len({gt.ground_truth_id for gt in db.inserted}), 25)

    def test_no_write_when_every_row_is_rejected(self):
        db = _RecordingDB()
        core_dataset.curate_records_to_dataset(
            dataset_name="batched",
            records=pd.DataFrame([{"record_id": "r1", "input": None}]),
            db=db,
            mapping=core_dataset.TraceDatasetMapping(
                query="input", query_id="record_id"
            ),
            on_error="collect",
        )
        self.assertEqual(db.batch_sizes, [])

    def test_rows_are_curated_lazily(self):
        # Peak memory is one batch: rows must not all be materialized before
        # the first write.
        db = _RecordingDB()
        seen = []

        core_dataset.curate_records_to_dataset(
            dataset_name="batched",
            records=self.make_records(6),
            db=db,
            mapping=core_dataset.TraceDatasetMapping(
                query="input", query_id="record_id"
            ),
            expected_response_fn=lambda row: (
                seen.append((row["record_id"], len(db.batch_sizes))) or "ok"
            ),
            batch_size=2,
        )

        # The 5th row is only visited after two batches have been written.
        self.assertEqual(seen[4][1], 2)


class TestRecordIdInput(CurationTestCase):
    def curate_with_resolver(
        self, records, resolver, mapping_override=None, **kwargs
    ):
        return core_dataset.curate_records_to_dataset(
            dataset_name="production-failures",
            records=records,
            db=self.db,
            mapping=mapping_override
            if mapping_override is not None
            else MAPPING,
            record_resolver=resolver,
            **kwargs,
        )

    def test_record_ids_are_resolved(self):
        # A review export that carries only record ids and corrections must be
        # joined against the recorded content before mapping.
        review = pd.DataFrame([
            {
                "record_id": "record-1",
                "corrected_output": "an evaluation library",
            }
        ])
        calls = []

        result = self.curate_with_resolver(
            review, lambda ids: calls.append(ids) or RECORDS
        )

        self.assertEqual(calls, [["record-1"]])
        self.assertEqual(result.accepted, 1)

        df = self.ground_truths()
        self.assertEqual(df.iloc[0]["query"], "what is trulens?")
        self.assertEqual(
            df.iloc[0]["expected_response"], "an evaluation library"
        )

    def test_export_columns_win_over_resolved_ones(self):
        review = pd.DataFrame([
            {
                "record_id": "record-1",
                "input": "an edited question",
                "corrected_output": "an evaluation library",
            }
        ])

        self.curate_with_resolver(review, lambda ids: RECORDS)

        self.assertEqual(
            self.ground_truths().iloc[0]["query"], "an edited question"
        )

    def test_resolution_is_skipped_when_nothing_is_missing(self):
        calls = []

        self.curate_with_resolver(
            RECORDS, lambda ids: calls.append(ids) or RECORDS
        )

        self.assertEqual(calls, [])

    def test_metric_columns_are_resolved_for_an_export_with_text(self):
        # A review export can carry the question text and still need metric
        # scores that only the database holds.
        review = pd.DataFrame([
            {
                "record_id": "record-1",
                "input": "what is trulens?",
                "corrected_output": "an evaluation library",
            }
        ])

        result = self.curate_with_resolver(review, lambda ids: RECORDS)

        self.assertEqual(result.accepted, 1)
        self.assertEqual(
            self.ground_truths().iloc[0]["meta"]["groundedness"], 0.2
        )

    def test_column_still_missing_after_resolution_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            self.curate_with_resolver(
                RECORDS,
                lambda ids: RECORDS,
                mapping_override=core_dataset.TraceDatasetMapping(
                    query="input", metadata={"score": "NotAMetric"}
                ),
            )
        self.assertIn("NotAMetric", str(caught.exception))

    def test_resolver_failure_is_reported_clearly(self):
        review = pd.DataFrame([{"record_id": "record-1"}])

        def boom(ids):
            raise NotImplementedError("`record_ids` is not supported")

        with self.assertRaises(ValueError) as caught:
            self.curate_with_resolver(review, boom)

        message = str(caught.exception)
        self.assertIn("input", message)
        self.assertIn("`record_ids` is not supported", message)
        self.assertIn("get_records_and_feedback", message)

    def test_unresolvable_ids_still_report_missing_columns(self):
        review = pd.DataFrame([{"record_id": "record-404"}])

        with self.assertRaises(ValueError) as caught:
            self.curate_with_resolver(review, lambda ids: pd.DataFrame())
        self.assertIn("input", str(caught.exception))

    def test_session_wires_the_resolver_to_get_records_and_feedback(self):
        review = pd.DataFrame([
            {
                "record_id": "record-1",
                "corrected_output": "an evaluation library",
            }
        ])

        with mock.patch.object(
            core_session.TruSession,
            "get_records_and_feedback",
            return_value=(RECORDS, []),
        ) as spy:
            result = self.curate(records=review)

        spy.assert_called_once_with(record_ids=["record-1"])
        self.assertEqual(result.accepted, 1)
        self.assertEqual(
            self.ground_truths().iloc[0]["query"], "what is trulens?"
        )


class TestCsvAndJsonInputs(CurationTestCase):
    def test_csv_export(self):
        path = os.path.join(self._tempdir.name, "review.csv")
        RECORDS.to_csv(path, index=False)

        result = self.curate(records=pd.read_csv(path))

        self.assertEqual(result.accepted, 2)
        self.assertEqual(len(self.ground_truths()), 2)

    def test_json_export(self):
        path = os.path.join(self._tempdir.name, "review.json")
        RECORDS.to_json(path, orient="records")

        result = self.curate(records=pd.read_json(path))

        self.assertEqual(result.accepted, 2)

    def test_csv_and_dataframe_produce_the_same_ground_truths(self):
        path = os.path.join(self._tempdir.name, "review.csv")
        RECORDS.to_csv(path, index=False)

        from_df = self.curate()
        from_csv = self.curate(records=pd.read_csv(path))

        self.assertEqual(
            sorted(from_df.ground_truth_ids),
            sorted(from_csv.ground_truth_ids),
        )


class TestCookbookFlow(CurationTestCase):
    """The loop the cookbook notebook walks, without the tracing machinery.

    Guards the narrative in `examples/expositional/use_cases/trace_to_dataset.ipynb`
    against drifting away from the API.
    """

    def test_records_to_ground_truth_metric(self):
        # 1. records as `get_records_and_feedback` returns them, scored by a
        #    metric that flags hedged answers.
        records = pd.DataFrame([
            {
                "record_id": "record-1",
                "app_name": "support-bot",
                "app_version": "v1",
                "input": "what is trulens?",
                "output": "TruLens is a library for evaluating LLM apps.",
                "Answered": 1.0,
            },
            {
                "record_id": "record-2",
                "app_name": "support-bot",
                "app_version": "v1",
                "input": "how do i install trulens?",
                "output": "I'm not sure, please check the docs.",
                "Answered": 0.0,
            },
        ])

        # 2. select the low-scoring rows in pandas.
        failures = records[records["Answered"] < 1.0].copy()
        self.assertEqual(len(failures), 1)

        # 3. write the corrected answer.
        failures["corrected_output"] = ["Run `pip install trulens`."]

        # 4. curate.
        result = self.session.curate_records_to_dataset(
            dataset_name="support-bot-regressions",
            records=failures,
            mapping=core_dataset.TraceDatasetMapping(
                query="input",
                query_id="record_id",
                expected_response="corrected_output",
                metadata={"answered": "Answered"},
            ),
            on_error="collect",
        )
        self.assertEqual((result.accepted, result.rejected), (1, 0))

        # 5. load it back and score an answer against it.
        ground_truth = self.session.get_ground_truth(
            dataset_name="support-bot-regressions"
        )
        expected = dict(
            zip(ground_truth["query"], ground_truth["expected_response"])
        )

        def matches_expected(prompt: str, response: str) -> float:
            want = expected.get(prompt.strip())
            if want is None:
                return float("nan")
            return float(response.strip().lower() == want.strip().lower())

        self.assertEqual(
            matches_expected(
                "how do i install trulens?", "Run `pip install trulens`."
            ),
            1.0,
        )
        self.assertEqual(
            matches_expected("how do i install trulens?", "no idea"), 0.0
        )

        # Provenance survives the whole trip.
        meta = ground_truth.iloc[0]["meta"]
        self.assertEqual(meta["source_record_id"], "record-2")
        self.assertEqual(meta["answered"], 0.0)


if __name__ == "__main__":
    unittest.main()

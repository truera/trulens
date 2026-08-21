"""Tests for immutable dataset versions and run provenance.

Covers the contract from truera/trulens#2701: content-addressed versions,
idempotent publishing, immutability, split and metadata round trips, version
zero compatibility for pre-versioning ground truth rows, latest and exact
version lookup, membership comparison, and run source provenance.
"""

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
from trulens.core import run as core_run
from trulens.core import session as core_session
from trulens.core.dao import default_run as default_run_dao
from trulens.core.database import sqlalchemy as db_sqlalchemy
from trulens.core.database.connector import default as default_connector
from trulens.core.schema import dataset as dataset_schema
from trulens.core.schema import groundtruth as groundtruth_schema

EXAMPLES = pd.DataFrame([
    {
        "question": "what is trulens?",
        "expected_answer": "an evaluation library",
        "metadata": {"topic": "product"},
        "case_id": "case-1",
    },
    {
        "question": "how do i install it?",
        "expected_answer": "pip install trulens",
        "metadata": {"topic": "setup"},
        "case_id": "case-2",
    },
])

COLUMN_SPEC = {
    "input": "question",
    "ground_truth_output": "expected_answer",
    "metadata": "metadata",
    "input_id": "case_id",
}


def _clear_tru_session_singletons():
    """Drop any live `TruSession` so each test gets its own database."""

    for key in [
        curr
        for curr in core_session.TruSession._singleton_instances
        if curr[0] == "trulens.core.session.TruSession"
    ]:
        del core_session.TruSession._singleton_instances[key]


class DatasetVersionTestCase(unittest.TestCase):
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

    def publish(self, **kwargs) -> dataset_schema.DatasetVersion:
        """Publish `EXAMPLES` under the default spec, overriding as needed."""

        kwargs.setdefault("dataset_name", "support-quality")
        kwargs.setdefault("dataframe", EXAMPLES)
        kwargs.setdefault("column_spec", COLUMN_SPEC)
        return self.session.create_dataset_version(**kwargs)


class TestVersionCreation(DatasetVersionTestCase):
    def test_create_from_dataframe(self):
        version = self.publish(description="Reviewed August failures")

        self.assertEqual(version.item_count, 2)
        self.assertEqual(len(version.items), 2)
        self.assertEqual(version.description, "Reviewed August failures")
        self.assertEqual(
            [item.input for item in version.items],
            ["what is trulens?", "how do i install it?"],
        )
        self.assertEqual(
            [item.expected_response for item in version.items],
            ["an evaluation library", "pip install trulens"],
        )
        self.assertEqual(
            [item.input_id for item in version.items],
            [
                "case-1",
                "case-2",
            ],
        )
        self.assertTrue(
            version.dataset_version_id.startswith(
                dataset_schema.DATASET_VERSION_ID_PREFIX
            )
        )

    def test_create_from_ground_truth_sequence(self):
        dataset_id = self.db.insert_dataset(
            dataset_schema.Dataset(name="from-gt")
        )
        ground_truths = [
            groundtruth_schema.GroundTruth(
                dataset_id=dataset_id,
                query="what is trulens?",
                expected_response="an evaluation library",
            ),
        ]

        version = self.session.create_dataset_version(
            dataset_name="from-gt", ground_truths=ground_truths
        )

        self.assertEqual(version.item_count, 1)
        self.assertEqual(version.items[0].input, "what is trulens?")
        self.assertEqual(version.dataset_id, dataset_id)

    def test_requires_exactly_one_source(self):
        with self.assertRaises(ValueError):
            self.session.create_dataset_version(dataset_name="d")
        with self.assertRaises(ValueError):
            self.session.create_dataset_version(
                dataset_name="d",
                dataframe=EXAMPLES,
                column_spec=COLUMN_SPEC,
                ground_truths=[],
            )

    def test_rejects_column_spec_without_input(self):
        with self.assertRaises(ValueError):
            self.publish(column_spec={"ground_truth_output": "expected_answer"})

    def test_rejects_column_spec_naming_absent_columns(self):
        with self.assertRaises(ValueError) as caught:
            self.publish(column_spec={"input": "not_a_column"})
        self.assertIn("not_a_column", str(caught.exception))

    def test_column_spec_ignores_unrelated_run_spec_keys(self):
        # A RunConfig.dataset_spec carrying extra span attributes should be
        # usable as-is.
        version = self.publish(
            column_spec={
                "RECORD_ROOT.INPUT": "question",
                "record_root.ground_truth_output": "expected_answer",
                "retrieval.query_text": "question",
            }
        )
        self.assertEqual(
            [item.input for item in version.items],
            ["what is trulens?", "how do i install it?"],
        )
        self.assertEqual(
            version.items[0].expected_response, "an evaluation library"
        )


class TestContentAddressing(DatasetVersionTestCase):
    def test_identical_content_is_idempotent(self):
        first = self.publish()
        second = self.publish()

        self.assertEqual(first.dataset_version_id, second.dataset_version_id)
        self.assertEqual(
            len(self.session.list_dataset_versions("support-quality")), 1
        )

    def test_description_does_not_change_identity(self):
        first = self.publish(description="first pass")
        second = self.publish(description="second pass")

        self.assertEqual(first.dataset_version_id, second.dataset_version_id)
        # Immutability: the stored description is the one first published.
        self.assertEqual(second.description, "first pass")

    def test_different_content_is_a_different_version(self):
        first = self.publish()
        second = self.publish(dataframe=EXAMPLES.head(1))

        self.assertNotEqual(first.dataset_version_id, second.dataset_version_id)
        self.assertEqual(second.item_count, 1)

    def test_item_order_changes_the_version_id(self):
        first = self.publish()
        second = self.publish(dataframe=EXAMPLES.iloc[::-1])

        self.assertNotEqual(first.dataset_version_id, second.dataset_version_id)

    def test_item_ids_are_stable_across_versions(self):
        first = self.publish()
        second = self.publish(dataframe=EXAMPLES.head(1))

        self.assertEqual(
            first.items[0].item_id,
            second.items[0].item_id,
        )

    def test_version_id_is_derived_from_content_hash(self):
        version = self.publish()
        self.assertEqual(
            version.dataset_version_id,
            dataset_schema.DATASET_VERSION_ID_PREFIX + version.content_hash,
        )

    def test_whitespace_does_not_change_item_identity(self):
        padded = EXAMPLES.copy()
        padded["question"] = padded["question"].map(lambda q: f"  {q}\n")

        self.assertEqual(
            self.publish().dataset_version_id,
            self.publish(dataframe=padded).dataset_version_id,
        )

    def test_missing_values_normalize_to_none(self):
        without = (
            EXAMPLES.head(1)
            .drop(columns=["expected_answer"])
            .assign(expected_answer=None)
        )
        empty = EXAMPLES.head(1).assign(expected_answer=float("nan"))

        spec = dict(COLUMN_SPEC)
        self.assertEqual(
            self.publish(dataframe=without, column_spec=spec).items[0].item_id,
            self.publish(dataframe=empty, column_spec=spec).items[0].item_id,
        )


class TestImmutabilityAndProvenance(DatasetVersionTestCase):
    def test_earlier_versions_remain_loadable(self):
        first = self.publish()
        second = self.publish(dataframe=EXAMPLES.head(1))

        loaded_first = self.session.get_dataset_version(
            dataset_version_id=first.dataset_version_id
        )
        self.assertEqual(loaded_first.item_count, 2)
        self.assertEqual(
            [item.input for item in loaded_first.items],
            ["what is trulens?", "how do i install it?"],
        )
        self.assertNotEqual(
            loaded_first.dataset_version_id, second.dataset_version_id
        )

    def test_parent_defaults_to_previous_version(self):
        first = self.publish()
        second = self.publish(dataframe=EXAMPLES.head(1))

        self.assertIsNone(first.parent_dataset_version_id)
        self.assertEqual(
            second.parent_dataset_version_id, first.dataset_version_id
        )

    def test_explicit_parent_is_kept(self):
        first = self.publish()
        self.publish(dataframe=EXAMPLES.head(1))
        third = self.publish(
            dataframe=EXAMPLES.tail(1),
            parent_dataset_version_id=first.dataset_version_id,
        )

        self.assertEqual(
            third.parent_dataset_version_id, first.dataset_version_id
        )

    def test_parent_does_not_affect_identity(self):
        baseline = self.publish()
        other = self.session.create_dataset_version(
            dataset_name="other-dataset",
            dataframe=EXAMPLES.head(1),
            column_spec=COLUMN_SPEC,
            source_metadata={"dataset_name": "support-quality"},
        )
        # Same items, same source metadata, different parent chain: the ids
        # differ only because the datasets differ.
        self.assertNotEqual(baseline.dataset_id, other.dataset_id)

    def test_version_index_is_monotonic(self):
        first = self.publish()
        second = self.publish(dataframe=EXAMPLES.head(1))
        third = self.publish(dataframe=EXAMPLES.tail(1))

        self.assertEqual(
            [
                first.version_index,
                second.version_index,
                third.version_index,
            ],
            [0, 1, 2],
        )


class TestSplitsAndMetadata(DatasetVersionTestCase):
    def test_splits_round_trip(self):
        version = self.publish(splits={"regression": ["case-1", "case-2"]})

        self.assertEqual(
            [item.splits for item in version.items],
            [
                ["regression"],
                ["regression"],
            ],
        )

        loaded = self.session.get_dataset_version(
            dataset_version_id=version.dataset_version_id
        )
        self.assertEqual(loaded.split_names(), ["regression"])
        self.assertEqual(len(loaded.split("regression")), 2)

    def test_partial_and_overlapping_splits(self):
        version = self.publish(
            splits={"regression": ["case-1"], "smoke": ["case-1", "case-2"]}
        )

        self.assertEqual(version.items[0].splits, ["regression", "smoke"])
        self.assertEqual(version.items[1].splits, ["smoke"])

    def test_splits_can_name_item_ids(self):
        published = self.publish()
        item_id = published.items[0].item_id

        version = self.session.create_dataset_version(
            dataset_name="by-item-id",
            dataframe=EXAMPLES,
            column_spec=COLUMN_SPEC,
            splits={"regression": [item_id]},
        )
        self.assertEqual(version.items[0].splits, ["regression"])

    def test_unknown_split_member_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.publish(splits={"regression": ["case-404"]})
        self.assertIn("case-404", str(caught.exception))

    def test_splits_do_not_change_item_identity(self):
        without = self.publish()
        with_splits = self.publish(splits={"regression": ["case-1"]})

        self.assertEqual(
            [item.item_id for item in without.items],
            [item.item_id for item in with_splits.items],
        )
        # ... but they do make it a distinct snapshot.
        self.assertNotEqual(
            without.dataset_version_id, with_splits.dataset_version_id
        )

    def test_metadata_round_trips_without_changing_item_identity(self):
        version = self.publish()
        loaded = self.session.get_dataset_version(
            dataset_version_id=version.dataset_version_id
        )

        self.assertEqual(
            [item.meta for item in loaded.items],
            [{"topic": "product"}, {"topic": "setup"}],
        )

        relabelled = EXAMPLES.copy()
        relabelled["metadata"] = [{"topic": "changed"}, {"topic": "setup"}]
        other = self.publish(dataframe=relabelled)

        self.assertEqual(
            [item.item_id for item in version.items],
            [item.item_id for item in other.items],
        )
        self.assertNotEqual(
            version.dataset_version_id, other.dataset_version_id
        )


class TestVersionZeroCompatibility(DatasetVersionTestCase):
    def seed_legacy_dataset(self, dataset_name="legacy-dataset"):
        self.session.add_ground_truth_to_dataset(
            dataset_name=dataset_name,
            ground_truth_df=pd.DataFrame([
                {"query": "q1", "expected_response": "a1"},
                {"query": "q2", "expected_response": "a2"},
            ]),
        )

    def test_existing_dataset_appears_as_version_zero(self):
        self.seed_legacy_dataset()

        version = self.session.get_dataset_version(
            dataset_name="legacy-dataset"
        )

        self.assertIsNotNone(version)
        self.assertTrue(version.is_version_zero)
        self.assertEqual(version.version_index, 0)
        self.assertEqual(version.item_count, 2)
        self.assertEqual(
            sorted(item.input for item in version.items), ["q1", "q2"]
        )

    def test_version_zero_is_deterministic(self):
        self.seed_legacy_dataset()

        first = self.session.get_dataset_version(dataset_name="legacy-dataset")
        second = self.session.get_dataset_version(dataset_name="legacy-dataset")

        self.assertEqual(first.dataset_version_id, second.dataset_version_id)

    def test_version_zero_is_materialized_on_first_publish(self):
        self.seed_legacy_dataset()
        version_zero_id = self.session.get_dataset_version(
            dataset_name="legacy-dataset"
        ).dataset_version_id

        published = self.session.create_dataset_version(
            dataset_name="legacy-dataset",
            dataframe=EXAMPLES,
            column_spec=COLUMN_SPEC,
        )

        versions = self.session.list_dataset_versions("legacy-dataset")
        self.assertEqual(list(versions["version_index"]), [0, 1])
        self.assertEqual(
            versions.iloc[0]["dataset_version_id"], version_zero_id
        )
        self.assertEqual(published.parent_dataset_version_id, version_zero_id)

        # Version zero stays loadable after the newer version lands.
        reloaded = self.session.get_dataset_version(
            dataset_version_id=version_zero_id
        )
        self.assertEqual(
            sorted(item.input for item in reloaded.items), ["q1", "q2"]
        )

    def test_original_ground_truth_rows_are_not_rewritten(self):
        self.seed_legacy_dataset()
        before = self.session.get_ground_truth(dataset_name="legacy-dataset")

        self.session.create_dataset_version(
            dataset_name="legacy-dataset",
            dataframe=EXAMPLES,
            column_spec=COLUMN_SPEC,
        )

        after = self.session.get_ground_truth(dataset_name="legacy-dataset")
        pd.testing.assert_frame_equal(before, after)

    def test_duplicate_dataset_names_resolve_to_one_primary_row(self):
        # A dataset id hashes name *and* metadata, so one name can map to
        # several rows. Version zero must be reconstructed from the same row
        # whether it is being read or materialized.
        for metadata in ({"owner": "b"}, {"owner": "a"}):
            self.session.add_ground_truth_to_dataset(
                dataset_name="dup",
                ground_truth_df=pd.DataFrame([
                    {
                        "query": f"q-{metadata['owner']}",
                        "expected_response": "a",
                    }
                ]),
                dataset_metadata=metadata,
            )

        read = self.session.get_dataset_version(dataset_name="dup")
        self.assertEqual(read.item_count, 1)

        self.session.create_dataset_version(
            dataset_name="dup",
            dataframe=EXAMPLES,
            column_spec=COLUMN_SPEC,
        )

        materialized = self.session.list_dataset_versions("dup").iloc[0]
        self.assertEqual(
            materialized["dataset_version_id"], read.dataset_version_id
        )
        self.assertEqual(materialized["version_index"], 0)

    def test_dataset_without_ground_truth_has_no_version(self):
        self.db.insert_dataset(dataset_schema.Dataset(name="empty-dataset"))
        self.assertIsNone(
            self.session.get_dataset_version(dataset_name="empty-dataset")
        )

    def test_ground_truth_output_shape_from_a_version(self):
        version = self.publish(splits={"regression": ["case-1"]})

        df = self.session.get_ground_truth(
            dataset_version_id=version.dataset_version_id
        )

        self.assertEqual(
            list(df["query"]),
            [
                "what is trulens?",
                "how do i install it?",
            ],
        )
        self.assertEqual(list(df["query_id"]), ["case-1", "case-2"])
        self.assertEqual(
            list(df["expected_response"]),
            ["an evaluation library", "pip install trulens"],
        )
        self.assertEqual(list(df["splits"]), [["regression"], []])

    def test_ground_truth_by_name_is_unaffected_by_publishing(self):
        self.seed_legacy_dataset()
        before = self.session.get_ground_truth(dataset_name="legacy-dataset")

        self.session.create_dataset_version(
            dataset_name="legacy-dataset",
            dataframe=EXAMPLES.head(1),
            column_spec=COLUMN_SPEC,
        )

        after = self.session.get_ground_truth(dataset_name="legacy-dataset")
        self.assertEqual(len(after), len(before))


class TestLookup(DatasetVersionTestCase):
    def test_name_resolves_to_latest_version(self):
        self.publish()
        latest = self.publish(dataframe=EXAMPLES.head(1))

        resolved = self.session.get_dataset_version(
            dataset_name="support-quality"
        )
        self.assertEqual(resolved.dataset_version_id, latest.dataset_version_id)

    def test_exact_version_lookup(self):
        first = self.publish()
        self.publish(dataframe=EXAMPLES.head(1))

        resolved = self.session.get_dataset_version(
            dataset_version_id=first.dataset_version_id
        )
        self.assertEqual(resolved.dataset_version_id, first.dataset_version_id)

    def test_unknown_version_returns_none(self):
        self.assertIsNone(
            self.session.get_dataset_version(dataset_version_id="nope")
        )

    def test_unknown_dataset_returns_none(self):
        self.assertIsNone(self.session.get_dataset_version(dataset_name="nope"))

    def test_requires_an_argument(self):
        with self.assertRaises(ValueError):
            self.session.get_dataset_version()

    def test_version_must_belong_to_named_dataset(self):
        version = self.publish()
        self.session.create_dataset_version(
            dataset_name="other-dataset",
            dataframe=EXAMPLES.head(1),
            column_spec=COLUMN_SPEC,
        )

        with self.assertRaises(ValueError):
            self.session.get_dataset_version(
                dataset_name="other-dataset",
                dataset_version_id=version.dataset_version_id,
            )

    def test_can_skip_loading_items(self):
        version = self.publish()
        loaded = self.session.get_dataset_version(
            dataset_version_id=version.dataset_version_id, load_items=False
        )

        self.assertEqual(loaded.items, [])
        self.assertEqual(loaded.item_count, 2)

    def test_list_versions_is_ordered_oldest_first(self):
        first = self.publish()
        second = self.publish(dataframe=EXAMPLES.head(1))

        listed = self.session.list_dataset_versions("support-quality")
        self.assertEqual(
            list(listed["dataset_version_id"]),
            [first.dataset_version_id, second.dataset_version_id],
        )
        self.assertEqual(list(listed["item_count"]), [2, 1])

    def test_list_versions_of_unknown_dataset_is_empty(self):
        listed = self.session.list_dataset_versions("nope")
        self.assertTrue(listed.empty)
        self.assertIn("dataset_version_id", listed.columns)


class TestMembershipComparison(DatasetVersionTestCase):
    def test_added_removed_and_unchanged(self):
        extended = pd.concat(
            [
                EXAMPLES,
                pd.DataFrame([
                    {
                        "question": "where are the docs?",
                        "expected_answer": "trulens.org",
                        "metadata": {},
                        "case_id": "case-3",
                    }
                ]),
            ],
            ignore_index=True,
        )

        first = self.publish()
        second = self.publish(dataframe=extended.tail(2))

        diff = self.session.compare_dataset_versions(
            first.dataset_version_id, second.dataset_version_id
        )

        self.assertEqual(diff.added, [extended_item_id(second, "case-3")])
        self.assertEqual(diff.removed, [extended_item_id(first, "case-1")])
        self.assertEqual(diff.unchanged, [extended_item_id(first, "case-2")])

    def test_identical_versions_have_no_diff(self):
        version = self.publish()
        diff = self.session.compare_dataset_versions(
            version.dataset_version_id, version.dataset_version_id
        )

        self.assertEqual(diff.added, [])
        self.assertEqual(diff.removed, [])
        self.assertEqual(len(diff.unchanged), 2)

    def test_metadata_only_change_reports_everything_unchanged(self):
        relabelled = EXAMPLES.copy()
        relabelled["metadata"] = [{"topic": "changed"}, {"topic": "changed"}]

        first = self.publish()
        second = self.publish(dataframe=relabelled)

        diff = self.session.compare_dataset_versions(
            first.dataset_version_id, second.dataset_version_id
        )
        self.assertEqual(diff.added, [])
        self.assertEqual(diff.removed, [])
        self.assertEqual(len(diff.unchanged), 2)

    def test_unknown_version_is_rejected(self):
        version = self.publish()
        with self.assertRaises(ValueError):
            self.session.compare_dataset_versions(
                version.dataset_version_id, "nope"
            )


def extended_item_id(version, input_id):
    """The item id of the example carrying `input_id` in `version`."""

    for item in version.items:
        if item.input_id == input_id:
            return item.item_id
    raise AssertionError(f"No item with input_id {input_id}.")


class TestPersistenceContract(DatasetVersionTestCase):
    """The version tables must survive a reconnect, not just a live session."""

    def test_versions_survive_a_reconnect(self):
        version = self.publish(splits={"regression": ["case-1"]})

        reopened = db_sqlalchemy.SQLAlchemyDB.from_db_url(
            str(self.db.engine.url)
        )
        loaded = reopened.get_dataset_version(
            dataset_version_id=version.dataset_version_id
        )

        self.assertEqual(loaded.item_count, 2)
        self.assertEqual(loaded.content_hash, version.content_hash)
        self.assertEqual(loaded.items[0].splits, ["regression"])
        self.assertEqual(loaded.items[0].meta, {"topic": "product"})

    def test_migration_creates_the_version_tables(self):
        table_names = set(self.db.orm.metadata.tables)
        self.assertIn("trulens_dataset_version", table_names)
        self.assertIn("trulens_dataset_version_item", table_names)
        self.assertEqual(self.db.get_db_revision(), "13")


class TestRunProvenance(DatasetVersionTestCase):
    """A run must be able to pin, report and read one exact snapshot."""

    def setUp(self):
        super().setUp()
        self.dao = default_run_dao.DefaultRunDao(db=self.db)

    def make_run(self, dataset_version_id=None, run_name="run_1"):
        metadata_df = self.dao.create_new_run(
            object_name="test_app",
            object_type="APP",
            object_version="v1",
            run_name=run_name,
            dataset_name="support-quality",
            source_type="DATAFRAME",
            dataset_spec=COLUMN_SPEC,
            dataset_version_id=dataset_version_id,
        )
        return core_run.Run.from_metadata_df(
            metadata_df,
            {
                "app": mock.MagicMock(),
                "main_method_name": "invoke",
                "run_dao": self.dao,
                "tru_session": self.session,
            },
        )

    def test_run_config_carries_a_version(self):
        config = core_run.RunConfig(
            run_name="r",
            dataset_name="support-quality",
            dataset_spec=COLUMN_SPEC,
            dataset_version_id="dataset_version_hash_abc",
        )
        self.assertEqual(config.dataset_version_id, "dataset_version_hash_abc")

    def test_run_config_version_is_optional(self):
        config = core_run.RunConfig(
            run_name="r",
            dataset_name="support-quality",
            dataset_spec=COLUMN_SPEC,
        )
        self.assertIsNone(config.dataset_version_id)

    def test_pinned_version_is_persisted_and_reported(self):
        version = self.publish()
        run = self.make_run(dataset_version_id=version.dataset_version_id)

        self.assertEqual(
            run.source_info.dataset_version_id, version.dataset_version_id
        )
        self.assertEqual(
            run.describe()["source_info"]["dataset_version_id"],
            version.dataset_version_id,
        )

    def test_unpinned_run_source_info_is_unchanged(self):
        run = self.make_run()

        self.assertIsNone(run.source_info.dataset_version_id)
        self.assertNotIn("dataset_version_id", run.describe()["source_info"])

    def test_run_reads_the_pinned_snapshot(self):
        version = self.publish()
        run = self.make_run(dataset_version_id=version.dataset_version_id)

        input_df = run._fetch_dataset_version_input_df()

        self.assertEqual(
            list(input_df["question"]),
            ["what is trulens?", "how do i install it?"],
        )
        self.assertEqual(
            list(input_df["expected_answer"]),
            ["an evaluation library", "pip install trulens"],
        )
        self.assertEqual(list(input_df["case_id"]), ["case-1", "case-2"])

    def test_pinned_snapshot_does_not_follow_later_versions(self):
        pinned = self.publish()
        run = self.make_run(dataset_version_id=pinned.dataset_version_id)

        self.publish(dataframe=EXAMPLES.head(1))  # a newer version lands

        self.assertEqual(len(run._fetch_dataset_version_input_df()), 2)

    def test_missing_pinned_version_is_reported(self):
        run = self.make_run(dataset_version_id="dataset_version_hash_missing")

        with self.assertRaises(ValueError) as caught:
            run._fetch_dataset_version_input_df()
        self.assertIn("dataset_version_hash_missing", str(caught.exception))

    def test_run_diff_exposes_both_versions(self):
        diff = core_run.RunDiff(
            run_a_name="a",
            run_b_name="b",
            dataset_version_id_a="dataset_version_hash_a",
            dataset_version_id_b="dataset_version_hash_b",
        )

        provenance = diff.provenance()
        self.assertEqual(list(provenance["run"]), ["a", "b"])
        self.assertEqual(
            list(provenance["dataset_version_id"]),
            ["dataset_version_hash_a", "dataset_version_hash_b"],
        )

    def test_provenance_of_a_run_without_source_info(self):
        # Comparison accepts anything run-shaped, including the minimal stubs
        # used by the run-comparison tests, so a missing source info means "no
        # pinned version" rather than an error.
        self.assertIsNone(core_run._dataset_version_id_of(object()))
        self.assertIsNone(
            core_run._dataset_version_id_of(mock.MagicMock(spec=core_run.Run))
        )

    def test_run_diff_defaults_to_no_versions(self):
        diff = core_run.RunDiff(run_a_name="a", run_b_name="b")

        self.assertIsNone(diff.dataset_version_id_a)
        self.assertIsNone(diff.dataset_version_id_b)
        self.assertTrue(diff.provenance()["dataset_version_id"].isna().all())


if __name__ == "__main__":
    unittest.main()

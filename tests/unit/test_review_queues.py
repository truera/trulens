"""Tests for human review queues and persistence.

Covers the queue half of truera/trulens#2700: field validation, queue
idempotency, every item-state transition, conditional claiming under
concurrency, stale-claim recovery, missing targets, review supersession, and
deletion behaviour.
"""

import os
import tempfile
import threading
import time
import unittest

from trulens.core import session as core_session
from trulens.core.database import sqlalchemy as db_sqlalchemy
from trulens.core.database.connector import default as default_connector
from trulens.core.schema import review as review_schema
from trulens.core.schema.review import FailureType
from trulens.core.schema.review import ReviewItemState
from trulens.core.schema.review import ReviewTarget
from trulens.core.schema.review import ReviewTargetType
from trulens.core.schema.review import SelectionSnapshot
from trulens.core.schema.review import Verdict


def _clear_tru_session_singletons():
    for key in [
        curr
        for curr in core_session.TruSession._singleton_instances
        if curr[0] == "trulens.core.session.TruSession"
    ]:
        del core_session.TruSession._singleton_instances[key]


def target(record_id: str, priority: float = 0.0, reason: str = "test"):
    return ReviewTarget(
        target_id=record_id,
        selection=SelectionSnapshot(selection_reason=reason, priority=priority),
    )


class ReviewTestCase(unittest.TestCase):
    """Base case giving each test its own file-backed SQLite database."""

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)

        self.db_path = os.path.join(self._tempdir.name, "trulens.sqlite")
        self.db = db_sqlalchemy.SQLAlchemyDB.from_db_url(
            f"sqlite:///{self.db_path}"
        )
        self.db.migrate_database()

        _clear_tru_session_singletons()
        self.addCleanup(_clear_tru_session_singletons)
        self.session = core_session.TruSession(
            connector=default_connector.DefaultDBConnector(database=self.db)
        )

    def make_queue(self, targets=None, **kwargs):
        kwargs.setdefault("name", "low-groundedness")
        return self.session.create_review_queue(
            targets=targets
            if targets is not None
            else [target("r1", 0.6), target("r2", 0.9)],
            **kwargs,
        )


class TestValidation(ReviewTestCase):
    def test_verdict_is_required_and_fixed(self):
        with self.assertRaises(ValueError) as caught:
            self.session.submit_human_review(target="r1", verdict="maybe")
        self.assertIn("pass, fail, needs_review", str(caught.exception))

    def test_failure_type_is_fixed(self):
        with self.assertRaises(ValueError) as caught:
            self.session.submit_human_review(
                target="r1", verdict="fail", failure_type="vibes"
            )
        self.assertIn("retrieval", str(caught.exception))

    def test_score_range_is_validated_before_persistence(self):
        for bad in (1.5, -0.1):
            with self.assertRaises(Exception):
                self.session.submit_human_review(
                    target="r1", verdict="pass", score=bad
                )
        self.assertTrue(self.session.get_human_reviews().empty)

    def test_score_bounds_are_inclusive(self):
        for good in (0.0, 1.0):
            self.session.submit_human_review(
                target=f"r{good}", verdict="pass", score=good
            )
        self.assertEqual(len(self.session.get_human_reviews()), 2)

    def test_enum_members_are_accepted_as_well_as_strings(self):
        review = self.session.submit_human_review(
            target="r1",
            verdict=Verdict.FAIL,
            failure_type=FailureType.PLANNING,
        )
        self.assertIs(review.verdict, Verdict.FAIL)
        self.assertIs(review.failure_type, FailureType.PLANNING)

    def test_queue_ordering_is_validated(self):
        with self.assertRaises(Exception):
            self.session.create_review_queue(name="q", order_by="whenever")

    def test_snapshot_priority_range(self):
        with self.assertRaises(Exception):
            SelectionSnapshot(selection_reason="x", priority=2.0)


class TestQueueMembership(ReviewTestCase):
    def test_targets_are_materialized(self):
        queue = self.make_queue()
        items = self.session.get_review_items(queue.review_queue_id)
        self.assertEqual(sorted(i.target_id for i in items), ["r1", "r2"])

    def test_adding_the_same_target_twice_is_idempotent(self):
        queue = self.make_queue()
        self.session.add_review_targets(
            queue.review_queue_id, [target("r1", 0.6)]
        )
        self.assertEqual(
            self.session.get_review_queue_progress(queue.review_queue_id)[
                "total"
            ],
            2,
        )

    def test_duplicate_targets_in_one_call_collapse(self):
        queue = self.make_queue(targets=[target("r1"), target("r1")])
        self.assertEqual(
            self.session.get_review_queue_progress(queue.review_queue_id)[
                "total"
            ],
            1,
        )

    def test_membership_is_stable(self):
        queue = self.make_queue()
        before = [
            i.review_item_id
            for i in self.session.get_review_items(queue.review_queue_id)
        ]
        self.session.add_review_targets(
            queue.review_queue_id, [target("r1"), target("r2")]
        )
        after = [
            i.review_item_id
            for i in self.session.get_review_items(queue.review_queue_id)
        ]
        self.assertEqual(before, after)

    def test_selection_snapshot_is_persisted(self):
        queue = self.make_queue(
            targets=[target("r1", 0.6, reason="Groundedness < 0.5")]
        )
        item = self.session.get_review_items(queue.review_queue_id)[0]
        self.assertEqual(item.selection.selection_reason, "Groundedness < 0.5")
        self.assertEqual(item.priority, 0.6)

    def test_queue_lookup_by_name(self):
        queue = self.make_queue(name="by-name")
        found = self.session.get_review_queue(name="by-name")
        self.assertEqual(found.review_queue_id, queue.review_queue_id)

    def test_queues_dataframe(self):
        self.make_queue(name="a")
        self.make_queue(name="b")
        queues = self.session.get_review_queues()
        self.assertEqual(sorted(queues["name"]), ["a", "b"])


class TestClaiming(ReviewTestCase):
    def test_claim_takes_the_most_severe_first(self):
        queue = self.make_queue()
        item = self.session.claim_next_review_item(queue.review_queue_id)
        self.assertEqual(item.target_id, "r2")
        self.assertIs(item.state, ReviewItemState.IN_REVIEW)
        self.assertIsNotNone(item.claim_token)

    def test_created_ordering_uses_queue_order(self):
        queue = self.make_queue(name="fifo", order_by="created")
        item = self.session.claim_next_review_item(queue.review_queue_id)
        self.assertEqual(item.target_id, "r1")

    def test_claim_records_the_reviewer_label(self):
        queue = self.make_queue()
        item = self.session.claim_next_review_item(
            queue.review_queue_id, reviewer="josh"
        )
        self.assertEqual(item.claimed_by, "josh")

    def test_an_item_is_only_handed_out_once(self):
        queue = self.make_queue()
        first = self.session.claim_next_review_item(queue.review_queue_id)
        second = self.session.claim_next_review_item(queue.review_queue_id)
        self.assertNotEqual(first.target_id, second.target_id)

    def test_claim_returns_none_when_nothing_is_pending(self):
        queue = self.make_queue(targets=[target("r1")])
        self.session.claim_next_review_item(queue.review_queue_id)
        self.assertIsNone(
            self.session.claim_next_review_item(queue.review_queue_id)
        )

    def test_claim_on_unknown_queue_raises(self):
        with self.assertRaises(ValueError):
            self.session.claim_next_review_item("no-such-queue")

    def test_concurrent_claims_never_double_assign(self):
        # The conditional update is what prevents two callers from both
        # claiming one item.
        queue = self.make_queue(
            targets=[target(f"r{i}", i / 20) for i in range(10)]
        )
        claimed = []
        lock = threading.Lock()

        def worker():
            db = db_sqlalchemy.SQLAlchemyDB.from_db_url(
                f"sqlite:///{self.db_path}"
            )
            while True:
                item = db.claim_next_review_item(queue.review_queue_id)
                if item is None:
                    return
                with lock:
                    claimed.append(item.target_id)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(claimed), 10)
        self.assertEqual(len(set(claimed)), 10, "an item was claimed twice")


class TestStaleClaims(ReviewTestCase):
    def test_a_fresh_claim_is_not_reclaimable(self):
        queue = self.make_queue(
            targets=[target("r1")], name="q", stale_claim_seconds=600
        )
        self.session.claim_next_review_item(queue.review_queue_id)
        self.assertIsNone(
            self.session.claim_next_review_item(queue.review_queue_id)
        )

    def test_a_stale_claim_is_recovered_on_the_next_pull(self):
        queue = self.make_queue(
            targets=[target("r1")], name="q", stale_claim_seconds=0.0
        )
        first = self.session.claim_next_review_item(queue.review_queue_id)
        time.sleep(0.01)
        second = self.session.claim_next_review_item(queue.review_queue_id)

        self.assertIsNotNone(second)
        self.assertEqual(first.target_id, second.target_id)
        # A new claim, not the old one handed back.
        self.assertNotEqual(first.claim_token, second.claim_token)

    def test_recovery_requires_an_explicit_pull(self):
        queue = self.make_queue(
            targets=[target("r1")], name="q", stale_claim_seconds=0.0
        )
        self.session.claim_next_review_item(queue.review_queue_id)
        time.sleep(0.01)

        # Nothing reaps in the background: the item is still in_review until
        # somebody asks for work.
        progress = self.session.get_review_queue_progress(queue.review_queue_id)
        self.assertEqual(progress[ReviewItemState.IN_REVIEW.value], 1)

    def test_stale_recovery_respects_the_configured_timeout(self):
        queue = self.make_queue(
            targets=[target("r1")], name="q", stale_claim_seconds=1000.0
        )
        self.session.claim_next_review_item(queue.review_queue_id)
        self.assertIsNone(
            self.session.claim_next_review_item(queue.review_queue_id)
        )


class TestItemTransitions(ReviewTestCase):
    def claim_one(self):
        queue = self.make_queue(targets=[target("r1")])
        return queue, self.session.claim_next_review_item(queue.review_queue_id)

    def test_release_returns_the_item_to_pending(self):
        queue, item = self.claim_one()
        released = self.session.release_review_item(item)

        self.assertIs(released.state, ReviewItemState.PENDING)
        self.assertIsNone(released.claim_token)
        self.assertIsNone(released.claimed_at)
        self.assertIsNotNone(
            self.session.claim_next_review_item(queue.review_queue_id)
        )

    def test_skip_takes_the_item_out_of_circulation(self):
        queue, item = self.claim_one()
        skipped = self.session.skip_review_item(item)

        self.assertIs(skipped.state, ReviewItemState.SKIPPED)
        self.assertIsNone(
            self.session.claim_next_review_item(queue.review_queue_id)
        )

    def test_unavailable_keeps_a_missing_target_visible(self):
        queue, item = self.claim_one()
        marked = self.session.mark_review_item_unavailable(item)

        self.assertIs(marked.state, ReviewItemState.UNAVAILABLE)
        progress = self.session.get_review_queue_progress(queue.review_queue_id)
        # Still counted, not deleted.
        self.assertEqual(progress[ReviewItemState.UNAVAILABLE.value], 1)
        self.assertEqual(progress["total"], 1)

    def test_submitting_completes_the_item(self):
        queue, item = self.claim_one()
        review = self.session.submit_human_review(
            target=item.target, verdict="fail", review_item=item
        )

        completed = self.session.get_review_items(queue.review_queue_id)[0]
        self.assertIs(completed.state, ReviewItemState.COMPLETED)
        self.assertEqual(completed.current_review_id, review.human_review_id)

    def test_progress_counts_every_state(self):
        queue = self.make_queue(
            targets=[target("r1"), target("r2"), target("r3")]
        )
        a = self.session.claim_next_review_item(queue.review_queue_id)
        self.session.skip_review_item(a)
        b = self.session.claim_next_review_item(queue.review_queue_id)
        self.session.submit_human_review(
            target=b.target, verdict="pass", review_item=b
        )

        progress = self.session.get_review_queue_progress(queue.review_queue_id)
        self.assertEqual(progress[ReviewItemState.PENDING.value], 1)
        self.assertEqual(progress[ReviewItemState.SKIPPED.value], 1)
        self.assertEqual(progress[ReviewItemState.COMPLETED.value], 1)
        self.assertEqual(progress["total"], 3)

    def test_a_foreign_claim_token_is_refused(self):
        _, item = self.claim_one()
        with self.assertRaises(ValueError):
            self.session.release_review_item(item, claim_token="not-my-token")

    def test_updating_an_unknown_item_raises(self):
        with self.assertRaises(ValueError):
            self.session.skip_review_item("no-such-item")

    def test_items_can_be_filtered_by_state(self):
        queue, _ = self.claim_one()
        self.assertEqual(
            len(
                self.session.get_review_items(
                    queue.review_queue_id, state=ReviewItemState.IN_REVIEW
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                self.session.get_review_items(
                    queue.review_queue_id, state=ReviewItemState.PENDING
                )
            ),
            0,
        )


class TestReviewPersistence(ReviewTestCase):
    def test_direct_review_needs_no_queue(self):
        review = self.session.submit_human_review(
            target="r1", verdict="pass", reviewer="josh"
        )
        self.assertIsNone(review.review_queue_id)
        self.assertEqual(len(self.session.get_human_reviews()), 1)

    def test_direct_and_queued_reviews_share_a_model(self):
        queue = self.make_queue(targets=[target("r1")])
        item = self.session.claim_next_review_item(queue.review_queue_id)
        queued = self.session.submit_human_review(
            target=item.target, verdict="fail", review_item=item
        )
        direct = self.session.submit_human_review(target="r9", verdict="fail")

        self.assertEqual(type(queued), type(direct))
        self.assertEqual(queued.review_queue_id, queue.review_queue_id)
        reviews = self.session.get_human_reviews()
        self.assertEqual(len(reviews), 2)

    def test_all_review_fields_round_trip(self):
        self.session.submit_human_review(
            target="r1",
            verdict="fail",
            score=0.25,
            failure_type="retrieval",
            corrected_output="Use only the retrieved support policy.",
            notes="The answer relies on an unrelated policy page.",
            reviewer="josh",
        )
        row = self.session.get_human_reviews().iloc[0]

        self.assertEqual(row["verdict"], "fail")
        self.assertEqual(row["score"], 0.25)
        self.assertEqual(row["failure_type"], "retrieval")
        self.assertEqual(
            row["corrected_output"], "Use only the retrieved support policy."
        )
        self.assertEqual(
            row["notes"], "The answer relies on an unrelated policy page."
        )
        self.assertEqual(row["reviewer"], "josh")

    def test_editing_supersedes_without_losing_history(self):
        first = self.session.submit_human_review(
            target="r1", verdict="fail", reviewer="josh"
        )
        second = self.session.submit_human_review(
            target="r1", verdict="pass", reviewer="josh"
        )

        self.assertEqual(second.supersedes_id, first.human_review_id)

        history = self.session.get_human_reviews()
        self.assertEqual(len(history), 2)

        current = self.session.get_human_reviews(include_superseded=False)
        self.assertEqual(len(current), 1)
        self.assertEqual(current.iloc[0]["verdict"], "pass")

    def test_supersession_chains(self):
        self.session.submit_human_review(
            target="r1", verdict="fail", reviewer="josh"
        )
        self.session.submit_human_review(
            target="r1", verdict="needs_review", reviewer="josh"
        )
        third = self.session.submit_human_review(
            target="r1", verdict="pass", reviewer="josh"
        )

        self.assertEqual(len(self.session.get_human_reviews()), 3)
        current = self.session.get_human_reviews(include_superseded=False)
        self.assertEqual(len(current), 1)
        self.assertEqual(
            current.iloc[0]["human_review_id"], third.human_review_id
        )

    def test_reviewers_review_independently(self):
        josh = self.session.submit_human_review(
            target="r1", verdict="fail", reviewer="josh"
        )
        sam = self.session.submit_human_review(
            target="r1", verdict="pass", reviewer="sam"
        )

        # Neither supersedes the other.
        self.assertIsNone(josh.supersedes_id)
        self.assertIsNone(sam.supersedes_id)
        self.assertEqual(
            len(self.session.get_human_reviews(include_superseded=False)), 2
        )

    def test_latest_review_is_per_reviewer(self):
        self.session.submit_human_review(
            target="r1", verdict="fail", reviewer="josh"
        )
        self.session.submit_human_review(
            target="r1", verdict="pass", reviewer="josh"
        )
        latest = self.db.get_latest_human_review(
            ReviewTargetType.RECORD, "r1", reviewer="josh"
        )
        self.assertIs(latest.verdict, Verdict.PASS)

    def test_reviews_filter_by_queue_and_reviewer(self):
        queue = self.make_queue(targets=[target("r1")])
        item = self.session.claim_next_review_item(queue.review_queue_id)
        self.session.submit_human_review(
            target=item.target,
            verdict="fail",
            reviewer="josh",
            review_item=item,
        )
        self.session.submit_human_review(
            target="r9", verdict="pass", reviewer="sam"
        )

        self.assertEqual(
            len(
                self.session.get_human_reviews(
                    review_queue_id=queue.review_queue_id
                )
            ),
            1,
        )
        self.assertEqual(len(self.session.get_human_reviews(reviewer="sam")), 1)

    def test_reviews_export_as_csv_and_json(self):
        self.session.submit_human_review(target="r1", verdict="pass")
        reviews = self.session.get_human_reviews()

        self.assertIn("verdict", reviews.to_csv(index=False))
        self.assertIn("verdict", reviews.to_json(orient="records"))


class TestDeletion(ReviewTestCase):
    def test_deleting_a_queue_keeps_reviews(self):
        queue = self.make_queue(targets=[target("r1")])
        item = self.session.claim_next_review_item(queue.review_queue_id)
        self.session.submit_human_review(
            target=item.target, verdict="fail", review_item=item
        )

        self.session.delete_review_queue(queue.review_queue_id)

        self.assertIsNone(
            self.session.get_review_queue(review_queue_id=queue.review_queue_id)
        )
        self.assertEqual(len(self.session.get_human_reviews()), 1)

    def test_deleting_a_queue_removes_only_its_items(self):
        keep = self.make_queue(name="keep", targets=[target("r1")])
        drop = self.make_queue(name="drop", targets=[target("r2")])

        self.session.delete_review_queue(drop.review_queue_id)

        self.assertEqual(
            self.session.get_review_queue_progress(keep.review_queue_id)[
                "total"
            ],
            1,
        )


class TestDatabaseLifecycle(ReviewTestCase):
    def test_review_tables_exist_after_migration(self):
        import sqlalchemy as sa

        tables = sa.inspect(self.db.engine).get_table_names()
        for name in (
            "trulens_review_queues",
            "trulens_review_items",
            "trulens_human_reviews",
        ):
            self.assertIn(name, tables)

    def test_reset_database_clears_reviews(self):
        self.make_queue()
        self.session.submit_human_review(target="r1", verdict="pass")

        self.session.reset_database()

        self.assertTrue(self.session.get_review_queues().empty)
        self.assertTrue(self.session.get_human_reviews().empty)

    def test_migration_is_additive_for_existing_data(self):
        from trulens.core.schema import dataset as dataset_schema

        # Data written before the review tables existed is untouched by them.
        dataset_id = self.db.insert_dataset(
            dataset=dataset_schema.Dataset(name="pre-existing")
        )
        self.make_queue()
        self.assertEqual(
            self.db.insert_dataset(
                dataset=dataset_schema.Dataset(name="pre-existing")
            ),
            dataset_id,
        )


class TestUnsupportedBackend(unittest.TestCase):
    def test_a_backend_without_review_support_says_so(self):
        from trulens.core.database.base import DB

        # Borrow the stubs rather than subclassing DB, whose other abstract
        # methods are beside the point here.
        class MinimalDB:
            _review_unsupported = DB._review_unsupported
            insert_review_queue = DB.insert_review_queue
            claim_next_review_item = DB.claim_next_review_item

        backend = MinimalDB()

        with self.assertRaises(NotImplementedError) as caught:
            backend.insert_review_queue(review_schema.ReviewQueue(name="q"))
        self.assertIn("SQLAlchemy", str(caught.exception))

        with self.assertRaises(NotImplementedError):
            backend.claim_next_review_item("q")


if __name__ == "__main__":
    unittest.main()

"""Serializable human-review classes.

Models the fixed review record and the pull-based queues described in
truera/trulens#2700. The field set is deliberately closed: new fields or enum
values are a normal schema change rather than something callers configure.
"""

from __future__ import annotations

from enum import Enum
import logging
import time
from typing import Any, Dict, Hashable, Optional

import pydantic
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)

ReviewQueueID = str
"""Unique identifier of a [ReviewQueue][trulens.core.schema.review.ReviewQueue]."""

ReviewItemID = str
"""Unique identifier of a [ReviewItem][trulens.core.schema.review.ReviewItem]."""

HumanReviewID = str
"""Unique identifier of a [HumanReview][trulens.core.schema.review.HumanReview]."""


class Verdict(str, Enum):
    """The reviewer's decision about a target. Required on every review."""

    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"


class FailureType(str, Enum):
    """Where a failing target went wrong."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    TOOL = "tool"
    PLANNING = "planning"
    SAFETY = "safety"
    OTHER = "other"


class ReviewTargetType(str, Enum):
    """Kind of object a review points at."""

    RECORD = "record"
    TRACE = "trace"
    CONVERSATION = "conversation"
    SPAN = "span"
    RUN_ITEM = "run_item"


class ReviewItemState(str, Enum):
    """Where a queue item is in its lifecycle.

    `PENDING` items are claimable. A claim moves an item to `IN_REVIEW`;
    submitting completes it, releasing returns it to `PENDING`, and skipping or
    marking it unavailable takes it out of circulation without a review.
    """

    PENDING = "pending"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


TERMINAL_ITEM_STATES = (
    ReviewItemState.COMPLETED,
    ReviewItemState.SKIPPED,
    ReviewItemState.UNAVAILABLE,
)
"""Item states from which no further transition happens on its own."""


class SelectionSnapshot(serial_utils.SerialModel):
    """Why a target was queued, frozen at selection time.

    Recomputing the source metrics later must not change what a reviewer sees,
    so everything here is copied out of the records dataframe rather than
    looked up during review.
    """

    selection_reason: str
    """Human-readable reason, e.g. `Groundedness < 0.5` or `top 20 cost in USD`."""

    priority: float = 0.0
    """Normalized severity in `[0.0, 1.0]`, used for queue ordering."""

    metric_name: Optional[str] = None
    """Metric that triggered selection, when one did."""

    metric_value: Optional[float] = None
    """The metric's value on the selected record."""

    metric_direction: Optional[bool] = None
    """`True` when higher is better for that metric, as reported with the records."""

    latency: Optional[float] = None
    """Recorded latency in seconds, when relevant."""

    cost: Optional[float] = None
    """Recorded cost, when relevant."""

    cost_currency: Optional[str] = None
    """Currency of `cost`. Costs are never compared across currencies."""

    app_name: Optional[str] = None
    """App the record came from."""

    app_version: Optional[str] = None
    """App version the record came from."""

    ts: Optional[float] = None
    """Timestamp of the source record."""

    @pydantic.field_validator("priority")
    @classmethod
    def _priority_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"`priority` must be within [0.0, 1.0], got {value}."
            )
        return value


class ReviewTarget(serial_utils.SerialModel, Hashable):
    """A typed reference to something that can be reviewed.

    The target is held by id only; the trace itself is loaded live during
    review so that a queue never carries a stale copy of a record.
    """

    target_type: ReviewTargetType = ReviewTargetType.RECORD
    """Kind of object referenced."""

    target_id: str
    """Id of the referenced object."""

    selection: Optional[SelectionSnapshot] = None
    """Why this target was selected, when it came from a selection."""

    def __hash__(self):
        return hash((self.target_type, self.target_id))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ReviewTarget):
            return NotImplemented
        return (
            self.target_type == other.target_type
            and self.target_id == other.target_id
        )

    @property
    def key(self) -> str:
        """Identity of the referenced object, ignoring why it was selected."""

        return f"{self.target_type.value}:{self.target_id}"


class HumanReview(serial_utils.SerialModel, Hashable):
    """One reviewer's decision about one target.

    Reviews are never edited in place: submitting again for the same target and
    reviewer writes a new row that points at the one it supersedes, so the full
    history is preserved.
    """

    human_review_id: HumanReviewID
    """Unique identifier for this review."""

    target_type: ReviewTargetType = ReviewTargetType.RECORD
    """Kind of object reviewed."""

    target_id: str
    """Id of the object reviewed."""

    verdict: Verdict
    """The reviewer's decision. Required."""

    score: Optional[float] = None
    """Normalized quality score within `[0.0, 1.0]`."""

    failure_type: Optional[FailureType] = None
    """Where a failing target went wrong."""

    corrected_output: Optional[str] = None
    """Reviewer-provided expected output or correction."""

    notes: Optional[str] = None
    """Free-form rationale."""

    reviewer: Optional[str] = None
    """Caller-supplied reviewer label. Not an authenticated identity."""

    supersedes_id: Optional[HumanReviewID] = None
    """The review this one replaces, when it is an edit."""

    review_queue_id: Optional[ReviewQueueID] = None
    """Queue the review came from, when it was not a direct review."""

    ts: float = pydantic.Field(default_factory=lambda: _now())
    """When the review was submitted."""

    def __init__(
        self,
        target_id: str,
        verdict: Verdict,
        target_type: ReviewTargetType = ReviewTargetType.RECORD,
        human_review_id: Optional[HumanReviewID] = None,
        **kwargs,
    ):
        kwargs["target_id"] = target_id
        kwargs["verdict"] = verdict
        kwargs["target_type"] = target_type

        super().__init__(human_review_id="temporary", **kwargs)

        if human_review_id is None:
            human_review_id = json_utils.obj_id_of_obj(
                json_utils.jsonify(self), prefix="human_review"
            )
        self.human_review_id = human_review_id

    @pydantic.field_validator("score")
    @classmethod
    def _score_in_range(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"`score` must be within [0.0, 1.0], got {value}.")
        return value

    def __hash__(self):
        return hash(self.human_review_id)

    @property
    def target(self) -> ReviewTarget:
        """The target this review is about."""

        return ReviewTarget(
            target_type=self.target_type, target_id=self.target_id
        )


class ReviewQueue(serial_utils.SerialModel, Hashable):
    """A fixed set of targets to work through."""

    review_queue_id: ReviewQueueID
    """Unique identifier for the queue."""

    name: str
    """Name of the queue."""

    description: Optional[str] = None
    """What the queue is for."""

    instructions: Optional[str] = None
    """Guidance shown to reviewers working the queue."""

    order_by: str = "severity"
    """How pending items are ordered when claimed. `severity` or `created`."""

    stale_claim_seconds: float = 900.0
    """How long a claim is honored before another caller may take the item."""

    created_at: float = pydantic.Field(default_factory=lambda: _now())
    """When the queue was created."""

    updated_at: float = pydantic.Field(default_factory=lambda: _now())
    """When the queue was last modified."""

    meta: types_schema.Metadata = pydantic.Field(default_factory=dict)
    """Metadata associated with the queue."""

    def __init__(
        self,
        name: str,
        review_queue_id: Optional[ReviewQueueID] = None,
        **kwargs,
    ):
        kwargs["name"] = name
        super().__init__(review_queue_id="temporary", **kwargs)

        if review_queue_id is None:
            review_queue_id = json_utils.obj_id_of_obj(
                json_utils.jsonify(self), prefix="review_queue"
            )
        self.review_queue_id = review_queue_id

    @pydantic.field_validator("order_by")
    @classmethod
    def _known_ordering(cls, value: str) -> str:
        if value not in ("severity", "created"):
            raise ValueError(
                f"`order_by` must be 'severity' or 'created', got {value!r}."
            )
        return value

    def __hash__(self):
        return hash(self.review_queue_id)


class ReviewItem(serial_utils.SerialModel, Hashable):
    """One target's place in a queue."""

    review_item_id: ReviewItemID
    """Unique identifier, derived from the queue and the target.

    Deriving it this way is what makes adding the same target to a queue twice
    idempotent."""

    review_queue_id: ReviewQueueID
    """Queue this item belongs to."""

    target_type: ReviewTargetType = ReviewTargetType.RECORD
    """Kind of object to review."""

    target_id: str
    """Id of the object to review."""

    priority: float = 0.0
    """Normalized severity used for ordering, copied from the selection."""

    state: ReviewItemState = ReviewItemState.PENDING
    """Where the item is in its lifecycle."""

    claim_token: Optional[str] = None
    """Token identifying the current claim, if any."""

    claimed_at: Optional[float] = None
    """When the current claim was taken."""

    claimed_by: Optional[str] = None
    """Caller label that holds the current claim."""

    current_review_id: Optional[HumanReviewID] = None
    """The most recent review submitted for this item."""

    selection: Optional[SelectionSnapshot] = None
    """Frozen reason this target was queued."""

    created_at: float = pydantic.Field(default_factory=lambda: _now())
    """When the item was added to the queue."""

    updated_at: float = pydantic.Field(default_factory=lambda: _now())
    """When the item last changed state."""

    def __init__(
        self,
        review_queue_id: ReviewQueueID,
        target_id: str,
        target_type: ReviewTargetType = ReviewTargetType.RECORD,
        review_item_id: Optional[ReviewItemID] = None,
        **kwargs,
    ):
        kwargs["review_queue_id"] = review_queue_id
        kwargs["target_id"] = target_id
        kwargs["target_type"] = target_type

        super().__init__(review_item_id="temporary", **kwargs)

        if review_item_id is None:
            review_item_id = json_utils.obj_id_of_obj(
                {
                    "review_queue_id": review_queue_id,
                    "target_type": target_type.value
                    if isinstance(target_type, ReviewTargetType)
                    else str(target_type),
                    "target_id": target_id,
                },
                prefix="review_item",
            )
        self.review_item_id = review_item_id

    def __hash__(self):
        return hash(self.review_item_id)

    @property
    def target(self) -> ReviewTarget:
        """The target this item points at, with its selection snapshot."""

        return ReviewTarget(
            target_type=self.target_type,
            target_id=self.target_id,
            selection=self.selection,
        )

    def is_claim_stale(self, stale_claim_seconds: float, now: float) -> bool:
        """Whether this item's claim has aged out and may be taken over."""

        if self.state is not ReviewItemState.IN_REVIEW:
            return False
        if self.claimed_at is None:
            return True
        return (now - self.claimed_at) >= stale_claim_seconds


def _now() -> float:
    """Current time as epoch seconds.

    Matches the float timestamps the runs tables already use.
    """

    return time.time()


def review_summary(reviews: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a review into the columns used for CSV/JSON export."""

    return {
        "human_review_id": reviews.get("human_review_id"),
        "target_type": reviews.get("target_type"),
        "target_id": reviews.get("target_id"),
        "verdict": reviews.get("verdict"),
        "score": reviews.get("score"),
        "failure_type": reviews.get("failure_type"),
        "corrected_output": reviews.get("corrected_output"),
        "notes": reviews.get("notes"),
        "reviewer": reviews.get("reviewer"),
        "supersedes_id": reviews.get("supersedes_id"),
        "review_queue_id": reviews.get("review_queue_id"),
        "ts": reviews.get("ts"),
    }


# HACK013: Need these if using __future__.annotations .
SelectionSnapshot.model_rebuild()
ReviewTarget.model_rebuild()
HumanReview.model_rebuild()
ReviewQueue.model_rebuild()
ReviewItem.model_rebuild()

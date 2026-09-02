"""DataFrame-first selection of traces for human review.

Chooses which records go into a [ReviewQueue][trulens.core.schema.review.ReviewQueue]
by filtering the dataframe that
[get_records_and_feedback][trulens.core.session.TruSession.get_records_and_feedback]
already returns, rather than introducing a database query language.

Every selected target freezes why it was chosen, so recomputing the source
metrics later never changes what a reviewer sees.
"""

from __future__ import annotations

import abc
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from trulens.core.schema import review as review_schema

logger = logging.getLogger(__name__)

LATENCY_COLUMN = "latency"
"""Records column holding latency in seconds."""

COST_COLUMN = "total_cost"
"""Records column holding cost."""

CURRENCY_COLUMN = "cost_currency"
"""Records column holding the currency of `total_cost`."""

RECORD_ID_COLUMN = "record_id"
"""Records column holding the record id."""

ORDER_BY_SEVERITY = "severity"
ORDER_BY_CREATED = "created"
ORDER_BY_MODES = (ORDER_BY_SEVERITY, ORDER_BY_CREATED)


def _direction_column(metric: str) -> str:
    """Name of the column carrying a metric's direction."""

    return f"{metric} direction"


def _is_missing(value: Any) -> bool:
    """Whether a dataframe value should be treated as absent."""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _require_column(records: pd.DataFrame, column: str, purpose: str) -> None:
    """Fail loudly when a predicate names a column the dataframe lacks.

    A typo that silently selected nothing would be worse than an error, so a
    missing column is reported with what is actually available.
    """

    if column in records.columns:
        return

    available = ", ".join(str(c) for c in records.columns)
    raise ValueError(
        f"Cannot select by {purpose}: the records dataframe has no column "
        f"{column!r}. Available columns: {available}."
    )


def _direction_of(records: pd.DataFrame, metric: str) -> Optional[bool]:
    """Read a metric's direction as reported alongside the records.

    Returns `None` when the records carry no direction for the metric, which
    callers treat as higher-is-better while warning.
    """

    column = _direction_column(metric)
    if column not in records.columns:
        return None

    values = [v for v in records[column] if not _is_missing(v)]
    if not values:
        return None

    directions = {bool(v) for v in values}
    if len(directions) > 1:
        logger.warning(
            "Metric %r reports more than one direction across the records; "
            "treating it as higher-is-better.",
            metric,
        )
        return True

    return directions.pop()


def _resolved_direction(records: pd.DataFrame, metric: str) -> bool:
    """A metric's direction, defaulting to higher-is-better with a warning."""

    direction = _direction_of(records, metric)
    if direction is None:
        logger.warning(
            "The records carry no direction for metric %r; defaulting to "
            "higher_is_better=True.",
            metric,
        )
        return True
    return direction


def _clamp(value: float) -> float:
    """Squash a severity into `[0.0, 1.0]`."""

    if _is_missing(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


class PredicateResult:
    """What one predicate found in a records dataframe.

    Carries a per-row mask alongside the reason, severity and snapshot fields
    that the matched rows should freeze.
    """

    def __init__(
        self,
        mask: pd.Series,
        reasons: pd.Series,
        priorities: pd.Series,
        fields: Optional[Dict[str, pd.Series]] = None,
    ):
        self.mask = mask
        self.reasons = reasons
        self.priorities = priorities
        self.fields = fields or {}


class ReviewPredicate(abc.ABC):
    """A condition over a records dataframe.

    Predicates compose with `&` and `|`. Composition merges the reasons and
    keeps the highest severity, so an item queued for several reasons is
    ordered by its worst one.
    """

    @abc.abstractmethod
    def evaluate(self, records: pd.DataFrame) -> PredicateResult:
        """Apply this predicate to `records`."""

    def __and__(self, other: "ReviewPredicate") -> "ReviewPredicate":
        return _Composite(self, other, how="and")

    def __or__(self, other: "ReviewPredicate") -> "ReviewPredicate":
        return _Composite(self, other, how="or")


class _Composite(ReviewPredicate):
    """Two predicates joined by `&` or `|`."""

    def __init__(self, left: ReviewPredicate, right: ReviewPredicate, how: str):
        self.left = left
        self.right = right
        self.how = how

    def evaluate(self, records: pd.DataFrame) -> PredicateResult:
        left = self.left.evaluate(records)
        right = self.right.evaluate(records)

        if self.how == "and":
            mask = left.mask & right.mask
            joiner = " and "
        else:
            mask = left.mask | right.mask
            joiner = " or "

        def _join(row_left: str, row_right: str) -> str:
            parts = [p for p in (row_left, row_right) if p]
            return joiner.join(parts)

        reasons = pd.Series(
            [
                _join(left.reasons.iloc[i], right.reasons.iloc[i])
                for i in range(len(records))
            ],
            index=records.index,
            dtype=object,
        )

        # The worst contributing reason drives ordering.
        priorities = pd.Series(
            [
                max(left.priorities.iloc[i], right.priorities.iloc[i])
                for i in range(len(records))
            ],
            index=records.index,
            dtype=float,
        )

        fields: Dict[str, pd.Series] = {}
        for source in (left.fields, right.fields):
            for key, series in source.items():
                if key not in fields:
                    fields[key] = series
                else:
                    # Prefer whichever side actually has a value per row.
                    existing = fields[key]
                    fields[key] = pd.Series(
                        [
                            existing.iloc[i]
                            if not _is_missing(existing.iloc[i])
                            else series.iloc[i]
                            for i in range(len(records))
                        ],
                        index=records.index,
                        dtype=object,
                    )

        return PredicateResult(mask, reasons, priorities, fields)


class _ScoreThreshold(ReviewPredicate):
    """Rows whose metric falls beyond a threshold."""

    def __init__(self, metric: str, threshold: float, below: bool):
        self.metric = metric
        self.threshold = threshold
        self.below = below

    def evaluate(self, records: pd.DataFrame) -> PredicateResult:
        _require_column(records, self.metric, f"metric {self.metric!r}")
        direction = _resolved_direction(records, self.metric)

        values = records[self.metric]
        comparison = "<" if self.below else ">"
        reason_text = f"{self.metric} {comparison} {self.threshold}"

        masks, reasons, priorities = [], [], []
        for value in values:
            if _is_missing(value):
                # NaN never matches: a missing score is not evidence of a
                # problem, and estimating one would be worse.
                masks.append(False)
                reasons.append("")
                priorities.append(0.0)
                continue

            value = float(value)
            matched = (
                value < self.threshold if self.below else value > self.threshold
            )
            masks.append(matched)
            reasons.append(reason_text if matched else "")
            priorities.append(self._priority(value) if matched else 0.0)

        return PredicateResult(
            mask=pd.Series(masks, index=records.index, dtype=bool),
            reasons=pd.Series(reasons, index=records.index, dtype=object),
            priorities=pd.Series(priorities, index=records.index, dtype=float),
            fields={
                "metric_name": pd.Series(
                    [self.metric if m else None for m in masks],
                    index=records.index,
                    dtype=object,
                ),
                "metric_value": pd.Series(
                    [
                        float(v) if m and not _is_missing(v) else None
                        for m, v in zip(masks, values)
                    ],
                    index=records.index,
                    dtype=object,
                ),
                "metric_direction": pd.Series(
                    [direction if m else None for m in masks],
                    index=records.index,
                    dtype=object,
                ),
            },
        )

    def _priority(self, value: float) -> float:
        """How far past the threshold a value sits, normalized."""

        if self.below:
            if self.threshold <= 0:
                return 1.0
            return _clamp((self.threshold - value) / self.threshold)
        span = 1.0 - self.threshold
        if span <= 0:
            return 1.0
        return _clamp((value - self.threshold) / span)


class _WorstScore(ReviewPredicate):
    """The `top_n` rows a metric rates worst, respecting its direction."""

    def __init__(self, metric: str, top_n: int):
        if top_n < 1:
            raise ValueError(f"`top_n` must be positive, got {top_n}.")
        self.metric = metric
        self.top_n = top_n

    def evaluate(self, records: pd.DataFrame) -> PredicateResult:
        _require_column(records, self.metric, f"metric {self.metric!r}")
        direction = _resolved_direction(records, self.metric)

        scored = [
            (i, float(v))
            for i, v in enumerate(records[self.metric])
            if not _is_missing(v)
        ]
        # Worst means lowest when higher is better, and highest otherwise.
        scored.sort(key=lambda pair: pair[1], reverse=not direction)
        chosen = scored[: self.top_n]
        ranks = {position: rank for rank, (position, _) in enumerate(chosen)}

        qualifier = "lowest" if direction else "highest"
        reason_text = f"top {self.top_n} {qualifier} {self.metric}"

        masks, reasons, priorities, values = [], [], [], []
        for position, value in enumerate(records[self.metric]):
            matched = position in ranks
            masks.append(matched)
            reasons.append(reason_text if matched else "")
            priorities.append(
                _clamp((self.top_n - ranks[position]) / self.top_n)
                if matched
                else 0.0
            )
            values.append(
                float(value) if matched and not _is_missing(value) else None
            )

        return PredicateResult(
            mask=pd.Series(masks, index=records.index, dtype=bool),
            reasons=pd.Series(reasons, index=records.index, dtype=object),
            priorities=pd.Series(priorities, index=records.index, dtype=float),
            fields={
                "metric_name": pd.Series(
                    [self.metric if m else None for m in masks],
                    index=records.index,
                    dtype=object,
                ),
                "metric_value": pd.Series(
                    values, index=records.index, dtype=object
                ),
                "metric_direction": pd.Series(
                    [direction if m else None for m in masks],
                    index=records.index,
                    dtype=object,
                ),
            },
        )


class _NumericThreshold(ReviewPredicate):
    """Rows whose numeric column exceeds a threshold."""

    def __init__(
        self,
        column: str,
        threshold: float,
        reason: str,
        field: str,
        purpose: str,
        currency: Optional[str] = None,
    ):
        self.column = column
        self.threshold = threshold
        self.reason = reason
        self.field = field
        self.purpose = purpose
        self.currency = currency

    def evaluate(self, records: pd.DataFrame) -> PredicateResult:
        _require_column(records, self.column, self.purpose)
        eligible = _currency_mask(records, self.currency, self.purpose)

        masks, reasons, priorities, values = [], [], [], []
        for position, value in enumerate(records[self.column]):
            if _is_missing(value) or not eligible[position]:
                # A missing value is never estimated, and a row in another
                # currency is never compared against this threshold.
                masks.append(False)
                reasons.append("")
                priorities.append(0.0)
                values.append(None)
                continue

            value = float(value)
            matched = value > self.threshold
            masks.append(matched)
            reasons.append(self.reason if matched else "")
            priorities.append(
                _clamp((value - self.threshold) / self.threshold)
                if matched and self.threshold > 0
                else (1.0 if matched else 0.0)
            )
            values.append(value if matched else None)

        fields = {
            self.field: pd.Series(values, index=records.index, dtype=object)
        }
        if self.currency is not None:
            fields["cost_currency"] = pd.Series(
                [self.currency if m else None for m in masks],
                index=records.index,
                dtype=object,
            )

        return PredicateResult(
            mask=pd.Series(masks, index=records.index, dtype=bool),
            reasons=pd.Series(reasons, index=records.index, dtype=object),
            priorities=pd.Series(priorities, index=records.index, dtype=float),
            fields=fields,
        )


class _TopNumeric(ReviewPredicate):
    """The `top_n` rows with the highest value in a numeric column."""

    def __init__(
        self,
        column: str,
        top_n: int,
        reason: str,
        field: str,
        purpose: str,
        currency: Optional[str] = None,
    ):
        if top_n < 1:
            raise ValueError(f"`top_n` must be positive, got {top_n}.")
        self.column = column
        self.top_n = top_n
        self.reason = reason
        self.field = field
        self.purpose = purpose
        self.currency = currency

    def evaluate(self, records: pd.DataFrame) -> PredicateResult:
        _require_column(records, self.column, self.purpose)
        eligible = _currency_mask(records, self.currency, self.purpose)

        scored = [
            (position, float(value))
            for position, value in enumerate(records[self.column])
            if not _is_missing(value) and eligible[position]
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        chosen = scored[: self.top_n]
        ranks = {position: rank for rank, (position, _) in enumerate(chosen)}

        masks, reasons, priorities, values = [], [], [], []
        for position, value in enumerate(records[self.column]):
            matched = position in ranks
            masks.append(matched)
            reasons.append(self.reason if matched else "")
            priorities.append(
                _clamp((self.top_n - ranks[position]) / self.top_n)
                if matched
                else 0.0
            )
            values.append(
                float(value) if matched and not _is_missing(value) else None
            )

        fields = {
            self.field: pd.Series(values, index=records.index, dtype=object)
        }
        if self.currency is not None:
            fields["cost_currency"] = pd.Series(
                [self.currency if m else None for m in masks],
                index=records.index,
                dtype=object,
            )

        return PredicateResult(
            mask=pd.Series(masks, index=records.index, dtype=bool),
            reasons=pd.Series(reasons, index=records.index, dtype=object),
            priorities=pd.Series(priorities, index=records.index, dtype=float),
            fields=fields,
        )


class _HasError(ReviewPredicate):
    """Rows whose recorded output is an error."""

    ERROR_COLUMNS = ("error", "record_error")

    def evaluate(self, records: pd.DataFrame) -> PredicateResult:
        column = next(
            (c for c in self.ERROR_COLUMNS if c in records.columns), None
        )

        if column is None:
            masks = [False] * len(records)
        else:
            masks = [not _is_missing(v) for v in records[column]]

        return PredicateResult(
            mask=pd.Series(masks, index=records.index, dtype=bool),
            reasons=pd.Series(
                ["record has an error" if m else "" for m in masks],
                index=records.index,
                dtype=object,
            ),
            priorities=pd.Series(
                # An error is always maximally severe.
                [1.0 if m else 0.0 for m in masks],
                index=records.index,
                dtype=float,
            ),
        )


def _currency_mask(
    records: pd.DataFrame, currency: Optional[str], purpose: str
) -> List[bool]:
    """Which rows are denominated in `currency`.

    Returns all-true when no currency is being filtered on. Costs are never
    compared across currencies, so a row whose currency is missing or different
    is simply not eligible.
    """

    if currency is None:
        return [True] * len(records)

    _require_column(records, CURRENCY_COLUMN, purpose)

    return [
        (not _is_missing(value)) and str(value) == currency
        for value in records[CURRENCY_COLUMN]
    ]


class ReviewTargets:
    """Builds review targets from a records dataframe.

    The class methods are predicate constructors; `from_records` applies them.

    Example:
        ```python
        targets = ReviewTargets.from_records(
            records_df,
            where=(
                ReviewTargets.low_score("Groundedness", below=0.5)
                | ReviewTargets.high_latency(above_seconds=8)
            ),
            order_by="severity",
            limit=100,
        )
        ```
    """

    @staticmethod
    def low_score(metric: str, below: float) -> ReviewPredicate:
        """Records scoring below `below` on `metric`."""

        return _ScoreThreshold(metric=metric, threshold=below, below=True)

    @staticmethod
    def high_score(metric: str, above: float) -> ReviewPredicate:
        """Records scoring above `above` on `metric`."""

        return _ScoreThreshold(metric=metric, threshold=above, below=False)

    @staticmethod
    def worst_score(metric: str, top_n: int) -> ReviewPredicate:
        """The `top_n` records a metric rates worst.

        Respects the direction reported with the records, so "worst" means the
        lowest scores for a higher-is-better metric and the highest for a
        lower-is-better one.
        """

        return _WorstScore(metric=metric, top_n=top_n)

    @staticmethod
    def high_latency(above_seconds: float) -> ReviewPredicate:
        """Records slower than `above_seconds`."""

        return _NumericThreshold(
            column=LATENCY_COLUMN,
            threshold=above_seconds,
            reason=f"latency > {above_seconds}s",
            field="latency",
            purpose="latency",
        )

    @staticmethod
    def slowest(top_n: int) -> ReviewPredicate:
        """The `top_n` slowest records."""

        return _TopNumeric(
            column=LATENCY_COLUMN,
            top_n=top_n,
            reason=f"top {top_n} latency",
            field="latency",
            purpose="latency",
        )

    @staticmethod
    def high_cost(above: float, currency: str) -> ReviewPredicate:
        """Records costing more than `above` in `currency`.

        `currency` is required: costs in different currencies are never
        compared, so rows denominated in anything else are ignored.
        """

        return _NumericThreshold(
            column=COST_COLUMN,
            threshold=above,
            reason=f"cost > {above} {currency}",
            field="cost",
            purpose="cost",
            currency=currency,
        )

    @staticmethod
    def most_expensive(top_n: int, currency: str) -> ReviewPredicate:
        """The `top_n` most expensive records in `currency`."""

        return _TopNumeric(
            column=COST_COLUMN,
            top_n=top_n,
            reason=f"top {top_n} cost in {currency}",
            field="cost",
            purpose="cost",
            currency=currency,
        )

    @staticmethod
    def has_error() -> ReviewPredicate:
        """Records that recorded an error."""

        return _HasError()

    @staticmethod
    def from_records(
        records: pd.DataFrame,
        where: Optional[ReviewPredicate] = None,
        order_by: str = ORDER_BY_SEVERITY,
        limit: Optional[int] = None,
        target_type: review_schema.ReviewTargetType = review_schema.ReviewTargetType.RECORD,
        id_column: str = RECORD_ID_COLUMN,
    ) -> List[review_schema.ReviewTarget]:
        """Select review targets from a records dataframe.

        Args:
            records: A dataframe as returned by `get_records_and_feedback()`.
            where: Which records to select. Selects every record when omitted.
            order_by: `"severity"` for worst-first, or `"created"` to keep the
                dataframe's own order.
            limit: Keep at most this many targets, after ordering.
            target_type: Kind of object the ids refer to.
            id_column: Column holding the target ids.

        Returns:
            Targets in queue order, each carrying a frozen
            [SelectionSnapshot][trulens.core.schema.review.SelectionSnapshot].

        Raises:
            ValueError: If `order_by` is unknown, `limit` is not positive, or a
                predicate names a column the dataframe does not have.
        """

        if order_by not in ORDER_BY_MODES:
            raise ValueError(
                f"`order_by` must be one of {ORDER_BY_MODES}, got {order_by!r}."
            )

        if limit is not None and limit < 1:
            raise ValueError(f"`limit` must be positive, got {limit}.")

        if not isinstance(records, pd.DataFrame):
            raise ValueError(
                f"`records` must be a pandas DataFrame, got "
                f"{type(records).__name__}."
            )

        _require_column(records, id_column, f"target id ({id_column})")

        if records.empty:
            return []

        if where is None:
            result = PredicateResult(
                mask=pd.Series(
                    [True] * len(records), index=records.index, dtype=bool
                ),
                reasons=pd.Series(
                    ["all records"] * len(records),
                    index=records.index,
                    dtype=object,
                ),
                priorities=pd.Series(
                    [0.0] * len(records), index=records.index, dtype=float
                ),
            )
        else:
            result = where.evaluate(records)

        selected: List[Tuple[int, review_schema.ReviewTarget]] = []
        for position in range(len(records)):
            if not bool(result.mask.iloc[position]):
                continue

            row = records.iloc[position]
            target_id = row[id_column]
            if _is_missing(target_id):
                logger.warning(
                    "Skipping a record with no %s.",
                    id_column,
                )
                continue

            snapshot = review_schema.SelectionSnapshot(
                selection_reason=str(result.reasons.iloc[position]),
                priority=float(result.priorities.iloc[position]),
                metric_name=_field(result, "metric_name", position),
                metric_value=_float_field(result, "metric_value", position),
                metric_direction=_bool_field(
                    result, "metric_direction", position
                ),
                latency=_float_field(result, "latency", position),
                cost=_float_field(result, "cost", position),
                cost_currency=_field(result, "cost_currency", position),
                app_name=_row_value(row, "app_name"),
                app_version=_row_value(row, "app_version"),
                ts=_row_float(row, "ts"),
            )

            selected.append((
                position,
                review_schema.ReviewTarget(
                    target_type=target_type,
                    target_id=str(target_id),
                    selection=snapshot,
                ),
            ))

        if order_by == ORDER_BY_SEVERITY:
            # Stable within equal severity so that the same dataframe always
            # produces the same queue.
            selected.sort(
                key=lambda pair: (-pair[1].selection.priority, pair[0])
            )

        targets = [target for _, target in selected]

        if limit is not None:
            targets = targets[:limit]

        return targets

    @staticmethod
    def preview(
        records: pd.DataFrame,
        where: Optional[ReviewPredicate] = None,
        order_by: str = ORDER_BY_SEVERITY,
        limit: Optional[int] = None,
        target_type: review_schema.ReviewTargetType = review_schema.ReviewTargetType.RECORD,
        id_column: str = RECORD_ID_COLUMN,
    ) -> pd.DataFrame:
        """Show what `from_records` would select, as a dataframe.

        Takes the same arguments and runs the same selection, so the ids
        previewed here are exactly the ids that get materialized into a queue.
        """

        targets = ReviewTargets.from_records(
            records,
            where=where,
            order_by=order_by,
            limit=limit,
            target_type=target_type,
            id_column=id_column,
        )

        return pd.DataFrame(
            data=[
                (
                    t.target_id,
                    t.target_type.value,
                    t.selection.selection_reason if t.selection else None,
                    t.selection.priority if t.selection else None,
                    t.selection.metric_name if t.selection else None,
                    t.selection.metric_value if t.selection else None,
                    t.selection.latency if t.selection else None,
                    t.selection.cost if t.selection else None,
                    t.selection.cost_currency if t.selection else None,
                )
                for t in targets
            ],
            columns=[
                "target_id",
                "target_type",
                "selection_reason",
                "priority",
                "metric_name",
                "metric_value",
                "latency",
                "cost",
                "cost_currency",
            ],
        )


def _field(result: PredicateResult, name: str, position: int) -> Optional[str]:
    """Read a snapshot field out of a predicate result."""

    series = result.fields.get(name)
    if series is None:
        return None
    value = series.iloc[position]
    return None if _is_missing(value) else str(value)


def _float_field(
    result: PredicateResult, name: str, position: int
) -> Optional[float]:
    series = result.fields.get(name)
    if series is None:
        return None
    value = series.iloc[position]
    return None if _is_missing(value) else float(value)


def _bool_field(
    result: PredicateResult, name: str, position: int
) -> Optional[bool]:
    series = result.fields.get(name)
    if series is None:
        return None
    value = series.iloc[position]
    return None if _is_missing(value) else bool(value)


def _row_value(row: pd.Series, column: str) -> Optional[str]:
    if column not in row.index:
        return None
    value = row[column]
    return None if _is_missing(value) else str(value)


def _row_float(row: pd.Series, column: str) -> Optional[float]:
    if column not in row.index:
        return None
    value = row[column]
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dedupe_targets(
    targets: Sequence[review_schema.ReviewTarget],
) -> List[review_schema.ReviewTarget]:
    """Drop repeated targets, keeping the first (most severe) occurrence."""

    seen = set()
    unique = []
    for target in targets:
        if target.key in seen:
            continue
        seen.add(target.key)
        unique.append(target)
    return unique

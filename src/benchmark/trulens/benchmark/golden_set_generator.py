"""Golden set generation from production TruLens records.

The hardest part of aligning LLM judges is building a good golden set. TruLens
already stores production records — inputs, outputs and judge scores — in its
database, so instead of hand-writing lists of dicts, users can sample real
traffic for human annotation.

[GoldenSetGenerator][trulens.benchmark.golden_set_generator.GoldenSetGenerator]
queries a `TruSession` for records (optionally filtered by app name, date range
or feedback score range), samples them with a configurable strategy (uniform
random, stratified across low/medium/high score buckets, or uncertainty — the
records whose judge scores sit closest to the decision boundary), and exports
the sample in the `GroundTruthAgreement` golden-set format or to CSV/JSON for
external annotation tools. Once annotated, the golden set can be validated,
loaded back, and persisted via `TruSession.add_ground_truth_to_dataset`.

Example:
    ```python
    from trulens.benchmark.golden_set_generator import GoldenSetGenerator
    from trulens.core.session import TruSession

    session = TruSession()
    generator = GoldenSetGenerator(session, seed=42)

    # Sample 50 records, stratified by existing relevance scores.
    sample = generator.sample(
        n=50,
        app_name="my_rag_app",
        strategy="stratified",
        feedback_name="relevance",
    )

    # Export for annotation.
    sample.to_csv("golden_set_to_annotate.csv")

    # After human annotation, validate, load back and persist.
    annotated = generator.load_annotations("golden_set_annotated.csv")
    generator.save_golden_set("my_golden_set", annotated)
    ```
"""

import json
import logging
import random
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

if TYPE_CHECKING:
    from trulens.core.session import TruSession

logger = logging.getLogger(__name__)

STRATEGY_RANDOM = "random"
STRATEGY_STRATIFIED = "stratified"
STRATEGY_UNCERTAINTY = "uncertainty"
STRATEGIES = (STRATEGY_RANDOM, STRATEGY_STRATIFIED, STRATEGY_UNCERTAINTY)

# Score bucket boundaries used by the stratified strategy.
STRATIFIED_BUCKETS: Sequence[Tuple[str, float, float]] = (
    ("low", 0.0, 1.0 / 3),
    ("medium", 1.0 / 3, 2.0 / 3),
    ("high", 2.0 / 3, 1.0),
)

# Columns of the golden-set annotation format, in export order. Provenance
# columns (record_id, app_name, ...) follow these when included.
GOLDEN_SET_COLUMNS = ("query", "expected_response", "expected_score")


def _as_text(value: Any) -> str:
    """Coerce a record input/output value to plain text.

    Record `input`/`output` values are stored either as raw strings, as
    JSON-encoded strings (e.g. `'"What is TruLens?"'`), or as already-decoded
    JSON objects depending on the database path. Unwrap the common
    JSON-encoded-string case and stringify everything else.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            try:
                loaded = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return value
            if isinstance(loaded, str):
                return loaded
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


class GoldenSetSample:
    """A sample of records ready for human annotation.

    Rows carry the golden-set columns (`query`, `expected_response` and an
    empty `expected_score` to be filled by an annotator) plus provenance
    columns (`record_id`, `app_name`, `app_version`, `ts`, and the sampled
    judge's name and score when a feedback function was used for sampling).

    Args:
        df: Sample rows, one per record.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def to_df(self, include_provenance: bool = True) -> pd.DataFrame:
        """Return the sample as a DataFrame.

        Args:
            include_provenance: Keep the provenance columns (record id, app
                name/version, timestamp, judge score). When `False`, only the
                golden-set columns are returned.

        Returns:
            A copy of the sample rows.
        """
        if include_provenance:
            return self.df.copy()
        return self.df.loc[:, list(GOLDEN_SET_COLUMNS)].copy()

    def to_list(self, include_provenance: bool = False) -> List[Dict]:
        """Return the sample as a list of golden-set dicts.

        The result matches the `GroundTruthAgreement` golden-set format:
        `[{"query": ..., "expected_response": ..., "expected_score": None}]`.
        """
        df = self.to_df(include_provenance=include_provenance)
        records = df.to_dict(orient="records")
        for record in records:
            if pd.isna(record.get("expected_score")):
                record["expected_score"] = None
        return records

    def to_csv(self, path: str, **kwargs) -> None:
        """Write the sample to a CSV file for external annotation.

        Args:
            path: Destination file path.
            **kwargs: Extra arguments passed to `pandas.DataFrame.to_csv`.
        """
        kwargs.setdefault("index", False)
        self.df.to_csv(path, **kwargs)

    def to_json(self, path: str) -> None:
        """Write the sample to a JSON file (list of row objects)."""
        records = self.to_list(include_provenance=True)
        for record in records:
            ts = record.get("ts")
            if ts is not None and not isinstance(ts, str):
                record["ts"] = "" if pd.isna(ts) else str(ts)
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)


class GoldenSetGenerator:
    """Samples production records from a `TruSession` for golden-set curation.

    Args:
        session: The `TruSession` whose database holds the records.
        seed: Optional seed making `sample` reproducible.
    """

    def __init__(self, session: "TruSession", seed: Optional[int] = None):
        self.session = session
        self._rng = random.Random(seed)

    def sample(
        self,
        n: int,
        app_name: Optional[str] = None,
        app_version: Optional[str] = None,
        strategy: str = STRATEGY_RANDOM,
        feedback_name: Optional[str] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        decision_boundary: float = 0.5,
        max_records: Optional[int] = None,
    ) -> GoldenSetSample:
        """Sample records for annotation.

        Args:
            n: Number of records to sample. If fewer records match the
                filters, all matching records are returned with a warning.
            app_name: Only sample records from this app.
            app_version: Only sample records from this app version.
            strategy: One of `"random"` (uniform), `"stratified"` (equal
                samples from the low/medium/high score buckets) or
                `"uncertainty"` (records whose scores are closest to
                `decision_boundary`).
            feedback_name: Feedback function whose scores drive the
                `"stratified"`/`"uncertainty"` strategies and the
                `min_score`/`max_score` filters. Required for those; ignored
                otherwise.
            start_time: Only sample records at or after this timestamp
                (anything `pandas.Timestamp` accepts).
            end_time: Only sample records at or before this timestamp.
            min_score: Only sample records with `feedback_name` score >= this.
            max_score: Only sample records with `feedback_name` score <= this.
            decision_boundary: Center of the `"uncertainty"` strategy;
                records are ranked by `abs(score - decision_boundary)`.
            max_records: Cap on how many records to fetch from the database
                before sampling.

        Returns:
            A [GoldenSetSample][trulens.benchmark.golden_set_generator.GoldenSetSample]
            with `expected_score` left empty for annotation.

        Raises:
            ValueError: On an unknown strategy, a missing/unknown
                `feedback_name` where one is required, or `n < 1`.
        """
        if n < 1:
            raise ValueError(f"`n` must be at least 1, got {n}.")
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Expected one of {STRATEGIES}."
            )

        needs_scores = (
            strategy in (STRATEGY_STRATIFIED, STRATEGY_UNCERTAINTY)
            or min_score is not None
            or max_score is not None
        )
        if needs_scores and not feedback_name:
            raise ValueError(
                "`feedback_name` is required for the "
                f"{STRATEGY_STRATIFIED!r}/{STRATEGY_UNCERTAINTY!r} strategies "
                "and for `min_score`/`max_score` filters."
            )

        records_df, feedback_names = self.session.get_records_and_feedback(
            app_name=app_name,
            app_version=app_version,
            limit=max_records,
        )

        if feedback_name and feedback_name not in records_df.columns:
            raise ValueError(
                f"Feedback function {feedback_name!r} not found in records. "
                f"Available: {sorted(feedback_names)}."
            )

        pool = self._build_pool(records_df, feedback_name)
        pool = self._apply_filters(
            pool,
            start_time=start_time,
            end_time=end_time,
            min_score=min_score,
            max_score=max_score,
            drop_unscored=needs_scores,
        )

        if pool.empty:
            logger.warning("No records matched the given filters.")
            return GoldenSetSample(pool)
        if len(pool) <= n:
            if len(pool) < n:
                logger.warning(
                    "Only %d records matched the filters; requested %d.",
                    len(pool),
                    n,
                )
            return GoldenSetSample(pool)

        if strategy == STRATEGY_RANDOM:
            positions = self._rng.sample(range(len(pool)), n)
        elif strategy == STRATEGY_STRATIFIED:
            positions = self._stratified_positions(pool, n)
        else:  # STRATEGY_UNCERTAINTY
            positions = self._uncertainty_positions(pool, n, decision_boundary)

        return GoldenSetSample(pool.iloc[positions])

    def load_annotations(
        self,
        source: Union[str, pd.DataFrame, Sequence[Dict]],
        require_scores: bool = True,
        score_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    ) -> pd.DataFrame:
        """Load and validate an annotated golden set.

        Args:
            source: Path to an annotated CSV or JSON file (as produced by
                [GoldenSetSample.to_csv][trulens.benchmark.golden_set_generator.GoldenSetSample.to_csv]
                /
                [GoldenSetSample.to_json][trulens.benchmark.golden_set_generator.GoldenSetSample.to_json]),
                or an equivalent DataFrame / list of dicts.
            require_scores: Require every row to carry a numeric
                `expected_score`.
            score_range: Inclusive `(low, high)` bounds scores must fall in;
                pass `None` to skip the range check.

        Returns:
            A DataFrame ready for
            [save_golden_set][trulens.benchmark.golden_set_generator.GoldenSetGenerator.save_golden_set],
            with `expected_score` (and `record_id` when present) also folded
            into a per-row `meta` dict so they survive persistence.

        Raises:
            ValueError: On missing columns, empty queries, missing scores
                (when required) or out-of-range scores.
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, str):
            if source.lower().endswith(".json"):
                with open(source) as f:
                    df = pd.DataFrame(json.load(f))
            else:
                df = pd.read_csv(source)
        else:
            df = pd.DataFrame(list(source))

        if "query" not in df.columns:
            raise ValueError("Annotations are missing the `query` column.")
        queries = df["query"].astype(str).str.strip()
        if df["query"].isna().any() or (queries == "").any():
            raise ValueError("Annotations contain rows with an empty query.")

        if "expected_score" not in df.columns:
            if require_scores:
                raise ValueError(
                    "Annotations are missing the `expected_score` column."
                )
            df["expected_score"] = None

        scores = pd.to_numeric(df["expected_score"], errors="coerce")
        if require_scores and scores.isna().any():
            missing = df.index[scores.isna()].tolist()
            raise ValueError(
                f"Rows {missing} have a missing or non-numeric "
                "`expected_score`. Annotate every row or pass "
                "`require_scores=False`."
            )
        if score_range is not None:
            low, high = score_range
            out_of_range = df.index[
                scores.notna() & ((scores < low) | (scores > high))
            ].tolist()
            if out_of_range:
                raise ValueError(
                    f"Rows {out_of_range} have `expected_score` outside "
                    f"[{low}, {high}]."
                )
        df["expected_score"] = scores

        if "expected_response" not in df.columns:
            df["expected_response"] = None

        def row_meta(row: pd.Series) -> Dict:
            meta: Dict[str, Any] = {}
            if not pd.isna(row["expected_score"]):
                meta["expected_score"] = float(row["expected_score"])
            record_id = row.get("record_id")
            if record_id is not None and not pd.isna(record_id):
                meta["record_id"] = record_id
            return meta

        df["meta"] = (
            df.apply(row_meta, axis=1)
            if not df.empty
            else pd.Series(dtype=object)
        )
        return df

    def save_golden_set(
        self,
        dataset_name: str,
        annotated: Union[str, pd.DataFrame, Sequence[Dict]],
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist an annotated golden set as a TruLens dataset.

        Args:
            dataset_name: Name of the dataset to create or extend.
            annotated: Annotated golden set — anything
                [load_annotations][trulens.benchmark.golden_set_generator.GoldenSetGenerator.load_annotations]
                accepts, or the DataFrame it returned.
            dataset_metadata: Optional metadata stored with the dataset.

        Returns:
            The number of ground truth rows written.
        """
        if isinstance(annotated, pd.DataFrame) and "meta" in annotated.columns:
            df = annotated
        else:
            df = self.load_annotations(annotated)
        self.session.add_ground_truth_to_dataset(
            dataset_name=dataset_name,
            ground_truth_df=df,
            dataset_metadata=dataset_metadata,
        )
        return len(df)

    def _build_pool(
        self, records_df: pd.DataFrame, feedback_name: Optional[str]
    ) -> pd.DataFrame:
        """Normalize raw records into annotation-ready rows."""
        if records_df.empty:
            columns = list(GOLDEN_SET_COLUMNS) + [
                "record_id",
                "app_name",
                "app_version",
                "ts",
            ]
            if feedback_name:
                columns += ["feedback_name", "feedback_score"]
            return pd.DataFrame(columns=columns)

        pool = pd.DataFrame({
            "query": records_df["input"].map(_as_text),
            "expected_response": records_df["output"].map(_as_text),
            "expected_score": None,
            "record_id": records_df.get("record_id"),
            "app_name": records_df.get("app_name"),
            "app_version": records_df.get("app_version"),
            "ts": pd.to_datetime(records_df.get("ts"), errors="coerce"),
        })
        if feedback_name:
            pool["feedback_name"] = feedback_name
            pool["feedback_score"] = pd.to_numeric(
                records_df[feedback_name], errors="coerce"
            )

        empty_queries = pool["query"].str.strip() == ""
        if empty_queries.any():
            logger.info(
                "Dropping %d records without a recorded input.",
                int(empty_queries.sum()),
            )
            pool = pool[~empty_queries]

        return pool.reset_index(drop=True)

    def _apply_filters(
        self,
        pool: pd.DataFrame,
        start_time: Optional[Any],
        end_time: Optional[Any],
        min_score: Optional[float],
        max_score: Optional[float],
        drop_unscored: bool,
    ) -> pd.DataFrame:
        """Apply date-range and score-range filters to the pool."""
        if pool.empty:
            return pool
        mask = pd.Series(True, index=pool.index)
        if start_time is not None:
            mask &= pool["ts"] >= pd.Timestamp(start_time)
        if end_time is not None:
            mask &= pool["ts"] <= pd.Timestamp(end_time)
        if drop_unscored:
            mask &= pool["feedback_score"].notna()
        if min_score is not None:
            mask &= pool["feedback_score"] >= min_score
        if max_score is not None:
            mask &= pool["feedback_score"] <= max_score
        return pool[mask].reset_index(drop=True)

    def _stratified_positions(self, pool: pd.DataFrame, n: int) -> List[int]:
        """Pick equal samples from the low/medium/high score buckets.

        When a bucket has fewer records than its share, the shortfall is
        backfilled uniformly from the remaining records.
        """
        scores = pool["feedback_score"]
        buckets: List[List[int]] = []
        for name, low, high in STRATIFIED_BUCKETS:
            last = name == STRATIFIED_BUCKETS[-1][0]
            in_bucket = (scores >= low) & (
                (scores <= high) if last else (scores < high)
            )
            buckets.append(list(pool.index[in_bucket]))

        base, remainder = divmod(n, len(buckets))
        # Buckets with the most records absorb the remainder first.
        by_size = sorted(
            range(len(buckets)), key=lambda i: len(buckets[i]), reverse=True
        )
        targets = [base] * len(buckets)
        for i in by_size[:remainder]:
            targets[i] += 1

        selected: List[int] = []
        leftover: List[int] = []
        for bucket, target in zip(buckets, targets):
            take = min(target, len(bucket))
            picked = set(self._rng.sample(bucket, take))
            selected.extend(sorted(picked))
            leftover.extend(i for i in bucket if i not in picked)

        shortfall = n - len(selected)
        if shortfall > 0 and leftover:
            selected.extend(
                self._rng.sample(leftover, min(shortfall, len(leftover)))
            )
        return selected

    def _uncertainty_positions(
        self, pool: pd.DataFrame, n: int, decision_boundary: float
    ) -> List[int]:
        """Pick the records whose scores are closest to the boundary."""
        distance = (pool["feedback_score"] - decision_boundary).abs()
        return list(distance.sort_values(kind="stable").index[:n])

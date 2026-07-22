"""Batch / offline evaluation of metrics over a pre-collected dataset.

This module provides [BatchEvaluator][trulens.core.batch.BatchEvaluator], a way
to run one or more [Metric][trulens.core.metric.metric.Metric]s over a tabular
dataset (a pandas DataFrame or a list of dicts) *without* a live app or a
recording session.

This is useful for:

- Running evaluations in CI against a golden test set.
- Benchmarking prompt or model changes across many examples.
- Evaluating historical production logs offline.

Instead of extracting metric inputs from trace spans, each metric's arguments
are mapped to dataset columns via
[Selector.from_column][trulens.core.feedback.selector.Selector.from_column].

Example:
    ```python
    import pandas as pd
    from trulens.core import BatchEvaluator, Metric, Selector
    from trulens.providers.openai import OpenAI

    provider = OpenAI()

    relevance = Metric(
        name="answer_relevance",
        implementation=provider.relevance,
        selectors={
            "prompt": Selector.from_column("query"),
            "response": Selector.from_column("answer"),
        },
    )

    evaluator = BatchEvaluator(metrics=[relevance])

    df = pd.DataFrame({
        "query": ["What is the capital of France?"],
        "answer": ["Paris is the capital of France."],
    })

    results = evaluator.evaluate(df)
    ```
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from trulens.core.feedback.selector import Selector
from trulens.core.metric.metric import Metric
from trulens.core.metric.metric import SkipEval
from trulens.core.utils import threading as threading_utils

logger = logging.getLogger(__name__)

# A single row of a dataset, as a mapping from column name to value.
Row = Mapping[str, Any]

# Accepted dataset input types.
DatasetLike = Union[pd.DataFrame, Sequence[Row]]


class BatchEvaluator:
    """Run metrics over a pre-collected dataset without a live app.

    A `BatchEvaluator` holds a list of [Metric][trulens.core.metric.metric.Metric]s
    whose selectors are created with
    [Selector.from_column][trulens.core.feedback.selector.Selector.from_column],
    mapping each metric argument to a column of the dataset.

    Args:
        metrics: The metrics to evaluate. Each metric's selectors must all be
            dataset (column) selectors created with `Selector.from_column`.

        max_workers: Maximum number of metric evaluations to run concurrently
            in a thread pool. Defaults to the thread pool's own default (based
            on the CPU count). Set to 1 to run serially. Provider-side rate
            limiting (requests per minute) is handled by the provider endpoints.
    """

    def __init__(
        self,
        metrics: Sequence[Metric],
        *,
        max_workers: Optional[int] = None,
    ):
        if not metrics:
            raise ValueError("`BatchEvaluator` requires at least one metric.")

        self.metrics: List[Metric] = list(metrics)
        self.max_workers = max_workers

        for metric in self.metrics:
            self._validate_metric(metric)

    @staticmethod
    def _validate_metric(metric: Metric) -> None:
        """Ensure a metric can be used in batch mode.

        All of a metric's selectors must be dataset (column) selectors.
        """
        if metric.imp is None:
            raise ValueError(
                f"Metric {metric.name!r} has no implementation to call."
            )
        if not metric.selectors:
            raise ValueError(
                f"Metric {metric.name!r} has no selectors. In batch mode, "
                "map each metric argument to a dataset column with "
                "`Selector.from_column(...)`."
            )
        for arg_name, selector in metric.selectors.items():
            if (
                not isinstance(selector, Selector)
                or not selector.is_dataset_selector
            ):
                raise ValueError(
                    f"Metric {metric.name!r} selector for argument "
                    f"{arg_name!r} is not a dataset selector. In batch mode, "
                    "use `Selector.from_column(column_name)` for every "
                    "argument."
                )

    def _unique_metric_names(self) -> List[str]:
        """Return metric names, disambiguating duplicates by suffixing an index."""
        names: List[str] = []
        seen: Dict[str, int] = {}
        for metric in self.metrics:
            name = metric.name
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[metric.name]}"
            else:
                seen[name] = 0
            names.append(name)
        return names

    @staticmethod
    def _normalize_rows(
        data: DatasetLike,
        column_map: Optional[Mapping[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize dataset input into a list of row dicts.

        Args:
            data: A pandas DataFrame or a sequence of dict-like rows.

            column_map: Optional mapping from dataset column name to the column
                name expected by the selectors. Renamed columns are added
                (the originals are kept as well).
        """
        if isinstance(data, pd.DataFrame):
            rows = data.to_dict(orient="records")
        elif isinstance(data, Mapping):
            raise TypeError(
                "`evaluate` expects a pandas DataFrame or a list of dicts, "
                "not a single mapping. Wrap a single row in a list."
            )
        else:
            rows = [dict(row) for row in data]

        if column_map:
            for row in rows:
                for src, dst in column_map.items():
                    if src in row:
                        row[dst] = row[src]

        return rows

    def _resolve_arg_values(
        self, metric: Metric, row: Row
    ) -> Dict[str, List[Any]]:
        """Resolve each metric argument to a list of candidate values for a row.

        A scalar column yields a single candidate. A list-valued column yields a
        single candidate (the whole list) when `collect_list` is True, or one
        candidate per element when `collect_list` is False.

        Raises:
            KeyError: If a required column is missing and the selector does not
                ignore missing values.
        """
        arg_values: Dict[str, List[Any]] = {}
        for arg_name, selector in metric.selectors.items():
            column = selector.dataset_column
            if column not in row:
                if selector.ignore_none_values:
                    raise SkipEval(
                        reason=f"column {column!r} missing from row",
                        metric=metric,
                    )
                raise KeyError(
                    f"Column {column!r} required by metric {metric.name!r} "
                    f"(argument {arg_name!r}) is not present in the dataset "
                    f"row. Available columns: {sorted(row.keys())}."
                )
            value = row[column]
            if value is None and selector.ignore_none_values:
                raise SkipEval(
                    reason=f"column {column!r} is None", metric=metric
                )
            if isinstance(value, list) and not selector.collect_list:
                arg_values[arg_name] = list(value)
            else:
                arg_values[arg_name] = [value]
        return arg_values

    def _evaluate_metric_on_row(
        self, metric: Metric, row: Row
    ) -> Tuple[Optional[float], Any, float]:
        """Evaluate a single metric on a single row.

        Returns:
            A tuple of (score, explanation, latency_seconds). The score is
            `None` when the evaluation was skipped, and `float('nan')` when the
            aggregate could not be computed.
        """
        start = time.time()
        try:
            arg_values = self._resolve_arg_values(metric, row)
        except SkipEval as e:
            logger.debug("Skipping metric %s: %s", metric.name, e)
            return None, {"skipped": str(e)}, time.time() - start

        arg_names = list(arg_values.keys())
        combinations = itertools.product(*(arg_values[a] for a in arg_names))

        scores: List[Any] = []
        explanations: List[Any] = []
        for combination in combinations:
            ins = dict(zip(arg_names, combination))
            try:
                result_and_meta = metric(**ins)
            except SkipEval as e:
                logger.debug("Skipping metric %s: %s", metric.name, e)
                continue

            if isinstance(result_and_meta, tuple):
                assert len(result_and_meta) == 2, (
                    "Metric functions must return either a single float, a "
                    "float-valued dict, or these together with a metadata dict "
                    "as a tuple."
                )
                score, meta = result_and_meta
            else:
                score, meta = result_and_meta, {}

            scores.append(score)
            explanations.append(meta)

        latency = time.time() - start

        if not scores:
            return None, {"skipped": "all combinations skipped"}, latency

        aggregate = self._aggregate_scores(metric, scores)
        explanation = (
            explanations[0] if len(explanations) == 1 else explanations
        )
        return aggregate, explanation, latency

    @staticmethod
    def _aggregate_scores(metric: Metric, scores: List[Any]) -> Any:
        """Aggregate one or more metric scores using the metric's aggregator."""
        if len(scores) == 1:
            return scores[0]
        agg = metric.agg if metric.agg is not None else np.mean
        try:
            return agg(scores)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Failed to aggregate scores for metric %s: %s", metric.name, e
            )
            return float("nan")

    def evaluate(
        self,
        data: DatasetLike,
        *,
        column_map: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """Evaluate all metrics over the dataset.

        Args:
            data: The dataset to evaluate, either a pandas DataFrame or a list
                of dict-like rows.

            column_map: Optional mapping from dataset column name to the column
                name referenced by the selectors, applied before evaluation.
                For example, ``{"user_question": "query"}`` makes a
                ``user_question`` column available as ``query``.

        Returns:
            A pandas DataFrame with one row per input row. The original columns
            are preserved, and for each metric ``M`` three columns are added:
            ``M`` (the score), ``M_explanation`` (metadata/reasons), and
            ``M_latency`` (evaluation time in seconds).
        """
        rows = self._normalize_rows(data, column_map=column_map)
        metric_names = self._unique_metric_names()

        # Pre-allocate the result grid so out-of-order completion is fine.
        n_rows = len(rows)
        scores: List[List[Any]] = [[None] * n_rows for _ in self.metrics]
        explanations: List[List[Any]] = [[None] * n_rows for _ in self.metrics]
        latencies: List[List[Any]] = [[None] * n_rows for _ in self.metrics]

        def work(
            metric_idx: int, row_idx: int
        ) -> Tuple[int, int, Optional[float], Any, float]:
            metric = self.metrics[metric_idx]
            score, explanation, latency = self._evaluate_metric_on_row(
                metric, rows[row_idx]
            )
            return metric_idx, row_idx, score, explanation, latency

        tasks = [
            (m_idx, r_idx)
            for m_idx in range(len(self.metrics))
            for r_idx in range(n_rows)
        ]

        if self.max_workers == 1 or n_rows == 0:
            results = [work(m_idx, r_idx) for m_idx, r_idx in tasks]
        else:
            with threading_utils.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [
                    executor.submit(work, m_idx, r_idx)
                    for m_idx, r_idx in tasks
                ]
                results = [future.result() for future in futures]

        for metric_idx, row_idx, score, explanation, latency in results:
            scores[metric_idx][row_idx] = score
            explanations[metric_idx][row_idx] = explanation
            latencies[metric_idx][row_idx] = latency

        result_df = pd.DataFrame(rows)
        for metric_idx, name in enumerate(metric_names):
            result_df[name] = scores[metric_idx]
            result_df[f"{name}_explanation"] = explanations[metric_idx]
            result_df[f"{name}_latency"] = latencies[metric_idx]

        return result_df

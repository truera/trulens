"""Convert between TruLens's Run/records data and EvalPort suites and result sets.

EvalPort (https://github.com/adhabnr-ux/evalport) is an open interchange format
(Apache 2.0) for portable LLM evaluation datasets: test cases, graders, suites,
and results as plain JSON, shared across evaluation tools (DeepEval, Promptfoo,
Inspect AI, AutoGen, CrewAI, Ragas, LangSmith, Braintrust, MLflow, Opik, and
now TruLens).

This module has two entry points, matching the shape used by every other
EvalPort adapter in the ecosystem:

    to_openeval(records_df, ...)
        Converts the DataFrame returned by ``Run.get_records()`` or
        ``Run.get_record_details()`` -- ``record_id``, ``input``, ``output``,
        ``latency``, plus one column per computed feedback score -- into an
        EvalPort ``ResultSet``, one ``GraderResult`` per feedback column.

    from_openeval(suite, ...)
        Converts an EvalPort suite's test cases into a pandas DataFrame plus
        a ready-to-use ``dataset_spec`` mapping, suitable to pass straight
        into ``Run.start(input_df=...)`` after constructing a
        ``RunConfig(dataset_spec=dataset_spec, ...)``.

Why a DataFrame boundary rather than the ``Run``/``RunConfig`` objects
directly: ``trulens.core.run.Run`` is a pydantic model that requires a live
``RunDaoBase``, ``TruSession`` and app instance just to construct -- it is
inherently tied to a running TruLens session with a backing database. The
DataFrames it hands back from ``get_records()``/``get_record_details()`` (and
accepts via ``start(input_df=...)``) are the actual portable surface, so this
module converts at that boundary rather than requiring a live session to
import or test it -- the same reason ``opik-openeval-adapter`` and
``ragas-openeval-adapter`` convert at their SDKs' plain-data boundaries
instead of their live-client objects.

Only the reserved TruLens dataset-spec fields this module explicitly maps
(``input``, ``ground_truth_output``, ``input_id``, ``record_id``, ``output``,
``latency``) round-trip through EvalPort's own TestCase/Result shape.
Everything else a *different* EvalPort-speaking tool would not know how to
interpret is preserved under ``metadata["trulens"]`` rather than silently
dropped -- the same lossiness tradeoff documented in every other adapter in
this ecosystem (see e.g. adapters/opik-openeval-adapter's README).
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except (
    ImportError
) as e:  # pragma: no cover - trulens-core already depends on pandas
    raise ImportError(
        "trulens-openeval requires pandas, which trulens-core already depends on."
    ) from e

try:
    from openeval.version import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval"]

# Columns Run.get_records() always includes that are not feedback scores.
_RECORD_OVERVIEW_COLUMNS = {"record_id", "input", "output", "latency"}

# TruLens pairs some feedback columns with a "<name>_calls" companion column
# holding the per-call detail (args/reason) for that feedback function. That
# companion is not itself a score and must never be treated as one.
_CALLS_SUFFIX = "_calls"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _feedback_column_to_grader_id(column_name: str) -> str:
    """Normalize a TruLens feedback name ('Context Relevance') into an
    EvalPort grader_id ('context_relevance')."""
    return str(column_name).strip().lower().replace(" ", "_")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_openeval(
    records_df: "pd.DataFrame",
    metric_columns: Optional[List[str]] = None,
    suite_id: str = "trulens_run",
    run_id: Optional[str] = None,
    pass_threshold: float = 0.5,
    latency_unit: str = "seconds",
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a completed Run's records into an EvalPort ``ResultSet``.

    Args:
        records_df: A DataFrame shaped like ``Run.get_records()``'s return
            value: ``record_id``, ``input``, ``output`` are required;
            ``latency`` and any number of feedback-score columns are
            optional. ``Run.get_record_details()``'s richer DataFrame also
            works, since it is a superset of the same columns.
        metric_columns: Which of ``records_df``'s columns are feedback
            scores. Defaults to every column that isn't one of
            ``record_id``/``input``/``output``/``latency`` and doesn't end in
            ``"_calls"`` (TruLens's per-call detail companion column, not a
            score itself).
        suite_id: The EvalPort ``ResultSet.suite_id`` this run's results
            belong to. TruLens records don't carry the id of the suite they
            were run against, so pass the value ``from_openeval`` returned
            when it built this run's ``input_df``, if applicable.
        run_id: The EvalPort ``ResultSet.run_id``. Defaults to
            ``records_df.attrs.get("run_name")`` (set this yourself, e.g.
            ``records_df.attrs["run_name"] = run.run_name``, since
            ``get_records()`` does not stamp it) and falls back to
            ``"trulens_run"`` if absent.
        pass_threshold: TruLens feedback scores are bare floats (typically
            0.0-1.0) with no built-in pass/fail. A grader result passes when
            ``score >= pass_threshold``. A result's overall ``passed``
            follows EvalPort's own convention: every one of its grader
            results must individually pass.
        latency_unit: Unit of the ``latency`` column, if present -- TruLens
            reports it in seconds by default. Set to ``"ms"`` if your
            source already reports milliseconds.
        started_at / completed_at: ISO-8601 timestamps for the ResultSet.
            ``started_at`` is required by the EvalPort schema; both default
            to the current time if omitted, since ``Run`` does not expose a
            run-level start/end timestamp through ``get_records()``.

    Returns:
        A dict matching EvalPort's ResultSet schema
        (validate with ``openeval.validate.validate_result_set``).

    Raises:
        ValueError: if ``records_df`` is empty or missing a required column.
    """
    if records_df is None or len(records_df) == 0:
        raise ValueError(
            "to_openeval: records_df is empty -- nothing to convert."
        )

    missing_required = [
        c
        for c in ("record_id", "input", "output")
        if c not in records_df.columns
    ]
    if missing_required:
        raise ValueError(
            "to_openeval: records_df is missing required column(s) "
            f"{missing_required}. Pass the DataFrame returned by "
            "Run.get_records() or Run.get_record_details(), not an "
            "arbitrary DataFrame."
        )

    if metric_columns is None:
        metric_columns = [
            c
            for c in records_df.columns
            if c not in _RECORD_OVERVIEW_COLUMNS
            and not str(c).endswith(_CALLS_SUFFIX)
        ]

    if run_id is None:
        attrs_run_name = getattr(records_df, "attrs", {}).get("run_name")
        run_id = str(attrs_run_name) if attrs_run_name else "trulens_run"

    results = []
    for _, row in records_df.iterrows():
        grader_results = []
        for col in metric_columns:
            if col not in records_df.columns:
                continue
            raw_score = row[col]
            if _is_missing(raw_score):
                continue
            score = max(0.0, min(1.0, float(raw_score)))
            grader_results.append({
                "grader_id": _feedback_column_to_grader_id(col),
                "type": "custom",
                "score": score,
                "passed": score >= pass_threshold,
            })

        result: Dict[str, Any] = {
            "test_case_id": str(row["record_id"]),
            "grader_results": grader_results,
            "passed": (
                all(g["passed"] for g in grader_results)
                if grader_results
                else False
            ),
            "metadata": {"trulens": {"record_id": str(row["record_id"])}},
        }

        if not _is_missing(row.get("output")):
            result["actual_output"] = str(row["output"])

        if "latency" in records_df.columns and not _is_missing(
            row.get("latency")
        ):
            latency_value = float(row["latency"])
            duration_ms = (
                latency_value * 1000
                if latency_unit == "seconds"
                else latency_value
            )
            result["duration_ms"] = max(0, round(duration_ms))

        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at or _now_iso(),
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total) if total else 0.0,
        },
    }
    result_set["completed_at"] = completed_at or _now_iso()

    return result_set


def from_openeval(
    suite: Dict[str, Any],
    input_id_column: str = "input_id",
) -> Tuple["pd.DataFrame", Dict[str, str]]:
    """Convert an EvalPort suite's test cases into a DataFrame plus a
    ``dataset_spec`` mapping ready for ``RunConfig``/``Run.start()``.

    Args:
        suite: An EvalPort suite dict (``openeval.validate.validate_suite``).
        input_id_column: Name of the DataFrame column that carries each
            EvalPort test case's ``id``, mapped to TruLens's reserved
            ``input_id`` dataset-spec field. TruLens does not reuse this
            value as its own ``record_id`` (that's assigned per invocation),
            but it is preserved so you can correlate a later ``to_openeval()``
            call's results back to the originating test case if you also
            propagate it yourself, e.g. by joining on this column.

    Returns:
        A ``(input_df, dataset_spec)`` tuple:

        - ``input_df``: a pandas DataFrame with one row per test case and
          columns ``input``, ``ground_truth_output``, and
          ``input_id_column``.
        - ``dataset_spec``: a dict mapping TruLens's reserved dataset-spec
          fields (as validated by
          ``trulens.core.run.validate_dataset_spec``) to ``input_df``'s own
          column names -- pass this straight into
          ``RunConfig(dataset_spec=dataset_spec, ...)``.

        Test cases whose ``expected_output`` is absent get ``None`` in the
        ``ground_truth_output`` column; TruLens's reference-free evaluators
        (e.g. groundedness, context relevance) don't require it, but
        reference-based ones will need it populated.

    Raises:
        ValueError: if the suite has no test cases.
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")

    rows = []
    for tc in test_cases:
        rows.append({
            "input": tc.get("input"),
            "ground_truth_output": tc.get("expected_output"),
            input_id_column: tc.get("id"),
        })

    input_df = pd.DataFrame(rows)
    dataset_spec = {
        "input": "input",
        "ground_truth_output": "ground_truth_output",
        "input_id": input_id_column,
    }
    return input_df, dataset_spec

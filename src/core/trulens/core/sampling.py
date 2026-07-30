"""Sampling configuration and controller for online evaluation.

Controls which records are evaluated during automatic post-ingest evaluation.
Does **not** affect explicit ``compute_metrics`` / ``compute_now`` calls.

.. note::
    Counters (throttle window, daily cost accumulator) are **per-process**.
    Multiple workers each enforce their own limits, so effective maximums
    scale with the number of processes.  The daily cost budget resets at
    **UTC midnight** and enforcement lags by roughly
    ``concurrency * cost_per_eval``, making it a **soft cap**.
"""

from __future__ import annotations

import collections
import contextvars
import datetime
from enum import Enum
import hashlib
import logging
import threading
from typing import Any, Dict, Optional, Union

import pydantic

logger = logging.getLogger(__name__)

# Context variable set by the evaluator on the ingest path only.
# When True, record_cost() calls in computer.py are allowed.
# When False (default / batch path), costs are not charged to the
# sampling budget so a backfill doesn't starve live evaluation.
ingest_eval_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ingest_eval_active", default=False
)


class EvalDecisionReason(str, Enum):
    """Why a record was or was not evaluated."""

    EVALUATED = "evaluated"
    """Record was sampled and evaluated normally."""

    NOT_CONFIGURED = "not_configured"
    """Sampling does not apply to this record (app not in per-app config).

    The record is evaluated normally — this is *not* a skip reason.
    """

    NOT_SAMPLED = "not_sampled"
    """Record was skipped by the probabilistic sampler."""

    THROTTLED = "throttled"
    """Record was skipped because the throttle window is saturated."""

    OVER_BUDGET = "over_budget"
    """Record was skipped because the daily cost budget is exhausted."""


class SamplingConfig(pydantic.BaseModel):
    """Immutable configuration for online-evaluation sampling.

    Attributes:
        sample_rate: Probability (0.0 -- 1.0) that a record is evaluated.
            Can also be a ``{app_name: rate}`` mapping for per-app rates.
            Default ``1.0`` (evaluate everything).
        throttle: Maximum evaluations per minute.  ``None`` means unlimited.
        cost_budget: Daily USD cap for evaluation cost.  ``None`` means
            unlimited.  Only enforceable for providers whose
            ``reports_costs`` property is ``True``.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    sample_rate: Union[float, Dict[str, float]] = 1.0
    """Probability of evaluating a record.

    * ``float`` -- a single global rate applied to every app.
    * ``dict[str, float]`` -- per-app rates keyed by app name.  Records
      from apps not in the dict are **not evaluated**.
    """

    throttle: Optional[int] = None
    """Max evaluations per minute (``None`` = unlimited)."""

    cost_budget: Optional[float] = None
    """Daily USD cap (``None`` = unlimited).

    Resets at UTC midnight.  Per-process, so N workers -> up to N * budget.
    """

    @pydantic.field_validator("sample_rate")
    @classmethod
    def _validate_sample_rate(cls, v):
        if isinstance(v, dict):
            for app_name, rate in v.items():
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(
                        f"sample_rate for app '{app_name}' must be "
                        f"between 0.0 and 1.0, got {rate}"
                    )
            return v
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"sample_rate must be between 0.0 and 1.0, got {v}"
            )
        return v

    @pydantic.field_validator("throttle")
    @classmethod
    def _validate_throttle(cls, v):
        if v is not None and v < 1:
            raise ValueError(
                f"throttle must be >= 1 (or None for unlimited), got {v}"
            )
        return v

    @pydantic.field_validator("cost_budget")
    @classmethod
    def _validate_cost_budget(cls, v):
        if v is not None and v <= 0:
            raise ValueError(
                f"cost_budget must be > 0 (or None for unlimited), got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# Deterministic sampling helper
# ---------------------------------------------------------------------------


def _hash_decision(record_id: str, app_name: Optional[str] = None) -> float:
    """Return a deterministic value in [0, 1) for a (record_id, app_name) pair.

    Uses SHA-256 so the decision is idempotent across retries, replays,
    and processes.  Salting with ``app_name`` avoids correlated subsets
    when per-app rates differ.
    """
    salt = app_name or ""
    digest = hashlib.sha256(f"{salt}:{record_id}".encode()).digest()
    # Use the first 8 bytes as an unsigned int, normalise to [0, 1).
    value = int.from_bytes(digest[:8], "big") / (2**64)
    return value


# ---------------------------------------------------------------------------
# SamplingController -- mutable, thread-safe, NOT a Pydantic model
# ---------------------------------------------------------------------------


class SamplingController:
    """Mutable state manager for sampling decisions.

    Holds the token-bucket (throttle), daily cost accumulator, and
    evaluated/skipped counters.  All public methods are thread-safe: the
    lock is held only for bookkeeping, never across an evaluation call.

    This class is designed to be substitutable in tests -- construct one
    directly and pass it wherever a controller is expected.
    """

    def __init__(self, config: SamplingConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

        # Throttle state: sliding window of evaluation timestamps.
        self._eval_timestamps: collections.deque[float] = collections.deque()

        # Daily cost state.
        self._daily_cost: float = 0.0
        self._daily_cost_date: datetime.date = datetime.datetime.now(
            tz=datetime.timezone.utc
        ).date()

        # Counters keyed by EvalDecisionReason.
        self._counters: Dict[EvalDecisionReason, int] = {
            r: 0 for r in EvalDecisionReason
        }

    # -- public API ---------------------------------------------------------

    @property
    def config(self) -> SamplingConfig:
        return self._config

    @property
    def counters(self) -> Dict[str, int]:
        """Return a snapshot of decision counters (reason -> count)."""
        with self._lock:
            return {r.value: c for r, c in self._counters.items()}

    def should_evaluate(
        self,
        record_id: str,
        app_name: Optional[str] = None,
    ) -> tuple:
        """Decide whether *record_id* should be evaluated.

        Order of checks: **app scope -> sample -> throttle -> budget**.

        Returns:
            ``(should_eval, meta)`` where *meta* is a dict with keys
            ``sample_rate``, ``eval_decision_reason``, and ``sampled``
            (a convenience boolean).
        """
        with self._lock:
            return self._should_evaluate_locked(record_id, app_name)

    def record_cost(self, cost: float) -> None:
        """Record evaluation cost for daily budget tracking."""
        with self._lock:
            self._maybe_reset_daily_cost()
            self._daily_cost += cost

    # -- internals ----------------------------------------------------------

    def _should_evaluate_locked(
        self,
        record_id: str,
        app_name: Optional[str],
    ) -> tuple:
        """Must be called with ``self._lock`` held."""

        rate = self._resolve_rate(app_name)

        # 1) App scope: if per-app rates are configured and this app
        #    is not in the dict, sampling does not apply — evaluate
        #    normally.  Configuring sampling on one app must NOT
        #    silently disable eval for every other app in the session.
        if rate is None:
            return self._decision(
                True,
                1.0,
                EvalDecisionReason.NOT_CONFIGURED,
            )

        # 2) Probabilistic sampling (deterministic via hash).
        h = _hash_decision(record_id, app_name)
        if h >= rate:
            return self._decision(
                False,
                rate,
                EvalDecisionReason.NOT_SAMPLED,
            )

        # 3) Throttle.
        if self._config.throttle is not None:
            now = datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
            cutoff = now - 60.0
            # Purge old entries.
            while self._eval_timestamps and self._eval_timestamps[0] < cutoff:
                self._eval_timestamps.popleft()
            if len(self._eval_timestamps) >= self._config.throttle:
                return self._decision(
                    False,
                    rate,
                    EvalDecisionReason.THROTTLED,
                )
            self._eval_timestamps.append(now)

        # 4) Cost budget.
        if self._config.cost_budget is not None:
            self._maybe_reset_daily_cost()
            if self._daily_cost >= self._config.cost_budget:
                return self._decision(
                    False,
                    rate,
                    EvalDecisionReason.OVER_BUDGET,
                )

        return self._decision(True, rate, EvalDecisionReason.EVALUATED)

    def _resolve_rate(self, app_name: Optional[str]) -> Optional[float]:
        """Return the sample rate for *app_name*, or None if out of scope."""
        sr = self._config.sample_rate
        if isinstance(sr, dict):
            if app_name is None:
                return None
            return sr.get(app_name)  # None -> app not in scope
        return sr

    def _maybe_reset_daily_cost(self) -> None:
        today = datetime.datetime.now(tz=datetime.timezone.utc).date()
        if today != self._daily_cost_date:
            self._daily_cost = 0.0
            self._daily_cost_date = today

    def _decision(
        self,
        should_eval: bool,
        rate: float,
        reason: EvalDecisionReason,
    ) -> tuple:
        self._counters[reason] += 1
        meta: Dict[str, Any] = {
            "sample_rate": rate,
            "eval_decision_reason": reason.value,
            "sampled": should_eval,
        }
        return should_eval, meta

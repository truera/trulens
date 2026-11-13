"""DSPy metric adapter for TruLens metrics.

DSPy evaluates and optimizes programs with a *metric*: a callable that takes a
gold `dspy.Example` and the program's `dspy.Prediction` and returns a score.
[TruDSPyMetric][trulens.apps.dspy.TruDSPyMetric] puts a TruLens
[Metric][trulens.core.Metric] behind that interface, reading each argument of
the metric implementation from an explicitly mapped example or prediction
field.

DSPy is not imported when this module loads. It is imported only to build the
`dspy.Prediction` that carries textual feedback to `dspy.GEPA`.
"""

from __future__ import annotations

import inspect
import logging
import math
import numbers
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from trulens.core.metric import metric as core_metric
from trulens.core.utils import constants as constants_utils

logger = logging.getLogger(__name__)

FieldSource = Union[str, Callable[[Any, Any], Any]]
"""Source of one implementation argument: a dotted path such as
`"prediction.answer"`, or a callable taking `(example, prediction)`."""

_ROOTS = ("example", "prediction")

# Metric.__call__ passes these to implementations that name them, so such
# parameters need no field mapping when the metric sets them.
_METRIC_SETTINGS = (
    "examples",
    "criteria",
    "additional_instructions",
    "min_score_val",
    "max_score_val",
    "temperature",
    "groundedness_configs",
    "enable_trace_compression",
)

_KEYWORD_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


class TruDSPyMetric:
    """Use a TruLens metric as a DSPy metric.

    DSPy calls a metric in three ways, and the return value follows the call:

    - `metric(example, prediction)`, as `dspy.Evaluate` and the scoring steps
      of optimizers such as MIPROv2 and GEPA call it, returns the score as a
      float.
    - `metric(example, prediction, trace)` with a trace, as bootstrap
      optimizers such as `BootstrapFewShot` call it to filter demos, returns
      `score >= threshold` when `threshold` is set and the float otherwise.
    - `metric(example, prediction, trace, pred_name, pred_trace)` with a
      predictor name, as `dspy.GEPA` calls it for feedback, returns
      `dspy.Prediction(score=..., feedback=...)` when the TruLens result
      carries reasons (as the `*_with_cot_reasons` provider methods do), and
      the float otherwise.

    A score that is not finite, or is the unparsable-score sentinel a judge
    reports when its reply held no score, raises `ValueError` instead of
    reaching DSPy as a verdict. DSPy counts a metric that raises as a failed
    example.

    Example:
        ```python
        from trulens.apps.dspy import TruDSPyMetric
        from trulens.core import Metric

        groundedness = TruDSPyMetric(
            Metric(
                implementation=provider.groundedness_measure_with_cot_reasons,
                name="Groundedness",
            ),
            fields={
                "source": "example.context",
                "statement": "prediction.answer",
            },
        )

        evaluate = dspy.Evaluate(devset=devset, metric=groundedness)
        ```

    Args:
        metric: The TruLens metric to wrap. A plain callable is wrapped as
            `Metric(implementation=metric)`. Arguments bound with
            [with_arguments][trulens.core.Metric.with_arguments] and metric
            settings such as `criteria` are passed on every call.
        fields: Maps implementation argument names to their sources. A source
            is either a dotted path that starts with `example.` or
            `prediction.`, such as `"prediction.answer"`, and may continue
            into nested attributes, or a callable that takes
            `(example, prediction)` and returns the value, for example to join
            a list of passages. Every required argument that is not bound on
            the metric must be mapped; nothing is inferred from argument
            names.
        threshold: When set, a call that carries a DSPy trace returns whether
            the score reaches this value instead of the score itself.

    Raises:
        TypeError: If `metric` is not callable, `fields` is not a mapping of
            names, or a source is neither a path nor a callable.
        ValueError: If `fields` names an argument the implementation does not
            take or that is already bound on the metric, holds a malformed
            path, or leaves a required argument unmapped.
    """

    def __init__(
        self,
        metric: Union[core_metric.Metric, Callable[..., Any]],
        *,
        fields: Mapping[str, FieldSource],
        threshold: Optional[float] = None,
    ) -> None:
        if not isinstance(metric, core_metric.Metric):
            if not callable(metric):
                raise TypeError(
                    "metric must be a trulens.core.Metric or a callable, not "
                    f"{type(metric).__name__}."
                )
            metric = core_metric.Metric(implementation=metric)
        if metric.imp is None:
            raise ValueError("metric has no implementation to call.")

        self._metric = metric
        self._fields = self._check_fields(fields)
        self._threshold = None if threshold is None else float(threshold)
        # Function-style name for code that labels metrics by `__name__`.
        self.__name__ = metric.name

    @property
    def name(self) -> str:
        """Name of the wrapped TruLens metric."""
        return self._metric.name

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Optional[Any] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[Any] = None,
    ) -> Any:
        """Score `prediction` against `example` with the TruLens metric.

        Args:
            example: The gold `dspy.Example`.
            prediction: The program's `dspy.Prediction`.
            trace: The DSPy trace, passed by optimizers while bootstrapping.
            pred_name: Name of the predictor `dspy.GEPA` wants feedback for.
            pred_trace: The predictor's sub-trace from `dspy.GEPA`. Unused.

        Returns:
            The score as a float; a bool when `threshold` is set and `trace`
                is given; or, when `pred_name` is given and the metric reports
                reasons, a `dspy.Prediction` with `score` and `feedback`.
        """
        kwargs = {
            arg: self._resolve(arg, source, example, prediction)
            for arg, source in self._fields.items()
        }
        score, meta = self._score(self._metric(**kwargs))

        if pred_name is not None:
            reasons = _format_reasons(meta)
            if not reasons:
                return score
            import dspy

            return dspy.Prediction(
                score=score,
                feedback=f"{self.name} score: {score:.2f}\n{reasons}",
            )
        if self._threshold is not None and trace is not None:
            return score >= self._threshold
        return score

    def _check_fields(
        self, fields: Mapping[str, FieldSource]
    ) -> Dict[str, FieldSource]:
        """Validate `fields` against the implementation signature."""
        if not isinstance(fields, Mapping) or not all(
            isinstance(arg, str) for arg in fields
        ):
            raise TypeError(
                "fields must map implementation argument names to DSPy field "
                "paths or callables."
            )

        for arg, source in fields.items():
            if callable(source):
                continue
            if not isinstance(source, str):
                raise TypeError(
                    f"Source for argument {arg!r} must be a path string or a "
                    f"callable, not {type(source).__name__}."
                )
            root, *attrs = source.split(".")
            if (
                root not in _ROOTS
                or not attrs
                or not all(attr.isidentifier() for attr in attrs)
            ):
                raise ValueError(
                    f"Path {source!r} for argument {arg!r} must look like "
                    "'example.<field>' or 'prediction.<field>'."
                )

        params = inspect.signature(self._metric.imp).parameters
        named = {n for n, p in params.items() if p.kind in _KEYWORD_KINDS}
        # Metric.__call__ forwards only named parameters, so a name that would
        # land in `**kwargs` is rejected too rather than silently dropped.
        unknown = sorted(set(fields) - named)
        if unknown:
            raise ValueError(
                f"{unknown} are not arguments of {self.name}; its arguments "
                f"are {sorted(named)}."
            )

        bound = self._metric.implementation_kwargs
        rebound = sorted(set(fields) & set(bound))
        if rebound:
            raise ValueError(
                f"{rebound} are already bound on {self.name} with "
                "with_arguments; remove them from fields."
            )

        supplied = set(fields) | set(bound)
        supplied.update(
            setting
            for setting in _METRIC_SETTINGS
            if getattr(self._metric, setting, None) is not None
        )
        unmapped = [
            n
            for n, p in params.items()
            if p.default is inspect.Parameter.empty
            and p.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            and n not in supplied
        ]
        if unmapped:
            raise ValueError(
                f"Required arguments {unmapped} of {self.name} are not mapped "
                "in fields."
            )

        return dict(fields)

    def _resolve(
        self, arg: str, source: FieldSource, example: Any, prediction: Any
    ) -> Any:
        """Read the value of implementation argument `arg` from DSPy data."""
        if callable(source):
            return source(example, prediction)

        root, *attrs = source.split(".")
        value = example if root == "example" else prediction
        for attr in attrs:
            try:
                value = getattr(value, attr)
            except AttributeError as exc:
                raise ValueError(
                    f"Argument {arg!r} of {self.name} maps to {source!r}, but "
                    f"the {root} has no field {attr!r}."
                ) from exc
        return value

    def _score(self, result: Any) -> Tuple[float, Dict[str, Any]]:
        """Reduce a TruLens result to one score and its metadata."""
        meta: Mapping[str, Any] = {}
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], Mapping)
        ):
            result, meta = result

        score = self._aggregate(result)
        if not math.isfinite(score) or constants_utils.is_unparsable_score(
            score
        ):
            raise ValueError(
                f"{self.name} produced no usable score ({score!r}). Non-finite "
                "scores and the unparsable-score sentinel "
                f"({constants_utils.UNPARSABLE_SCORE}) mark a failed "
                "evaluation, not a verdict."
            )
        return score, dict(meta)

    def _aggregate(self, value: Any) -> float:
        """Turn a score or a sequence of scores into one float.

        Sequences are reduced with the metric's `agg`, skipping
        unparsable-score sentinels first as `Metric.run` does.
        """
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if _is_number(value):
            return float(value)

        if (
            isinstance(value, (list, tuple))
            and value
            and all(_is_number(v) for v in value)
        ):
            scores = [
                float(v)
                for v in value
                if not constants_utils.is_unparsable_score(v)
            ]
            if len(scores) < len(value):
                logger.warning(
                    "%s: %d of %d scores were unparsable and are left out of "
                    "the aggregate.",
                    self.name,
                    len(value) - len(scores),
                    len(value),
                )
            if not scores:
                return constants_utils.UNPARSABLE_SCORE
            agg = self._metric.agg or np.mean
            aggregated = agg(np.array(scores))
            if _is_number(aggregated):
                return float(aggregated)
            value = aggregated

        raise ValueError(
            f"{self.name} returned {type(value).__name__} {value!r}, which "
            "cannot be reduced to one score. Return a number, a sequence of "
            "numbers, or a (score, metadata) tuple."
        )


def _is_number(value: Any) -> bool:
    """Whether `value` is a real or boolean scalar, numpy types included."""
    return isinstance(value, (numbers.Real, np.bool_))


def _format_reasons(meta: Mapping[str, Any]) -> str:
    """Render the reasons in TruLens metadata as text, or `""` if none."""
    reasons = meta.get("reasons") or meta.get("reason")
    return _render(reasons).strip() if reasons else ""


def _render(value: Any) -> str:
    if isinstance(value, Mapping):
        return "\n".join(f"{key}: {val}" for key, val in value.items())
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_render(item) for item in value)
    return str(value)

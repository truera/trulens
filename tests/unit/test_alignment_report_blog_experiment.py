"""Tests for the AlignmentReport blog experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.optional

pytest.importorskip("trulens.benchmark")

_SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "expositional"
    / "use_cases"
    / "alignment_report"
    / "alignment_report_before_after.py"
)


def _load_experiment_module():
    spec = importlib.util.spec_from_file_location(
        "alignment_report_before_after", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_frame() -> pd.DataFrame:
    rows = []
    relevance_patterns = [
        [1.0, 1.5, 2.5, 3.5, 4.5, 5.0],
        [1.25, 1.75, 2.75, 3.25, 4.25, 4.75],
    ]
    for article_index in range(20):
        pattern = relevance_patterns[article_index % len(relevance_patterns)]
        rows.append({
            "id": f"article-{article_index:02d}",
            "text": f"Source article {article_index}",
            "machine_summaries": np.asarray(
                [
                    f"Summary {article_index}-{summary_index}"
                    for summary_index in range(len(pattern))
                ],
                dtype=object,
            ),
            "relevance": np.asarray(pattern, dtype=float),
        })
    return pd.DataFrame(rows)


def test_select_examples_is_deterministic_balanced_and_article_disjoint():
    experiment = _load_experiment_module()

    first = experiment.select_examples(
        _source_frame(), samples_per_bucket=2, seed=20260731
    )
    second = experiment.select_examples(
        _source_frame(), samples_per_bucket=2, seed=20260731
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby(["split", "score_range"]).size().to_dict() == {
        ("development", "high"): 2,
        ("development", "low"): 2,
        ("development", "medium"): 2,
        ("held_out", "high"): 2,
        ("held_out", "low"): 2,
        ("held_out", "medium"): 2,
        ("validation", "high"): 2,
        ("validation", "low"): 2,
        ("validation", "medium"): 2,
    }
    split_article_ids = {
        split: set(rows["article_id"]) for split, rows in first.groupby("split")
    }
    for left, right in (
        ("development", "validation"),
        ("development", "held_out"),
        ("validation", "held_out"),
    ):
        assert split_article_ids[left].isdisjoint(split_article_ids[right])
    assert first["true_label"].between(0.0, 1.0).all()


def test_build_reports_uses_current_alignment_report_api():
    experiment = _load_experiment_module()
    examples = pd.DataFrame({
        "sample_id": [f"sample-{index}" for index in range(6)],
        "article_id": [f"article-{index}" for index in range(6)],
        "summary": [f"summary-{index}" for index in range(6)],
        "true_label": [0.0, 0.25, 0.5, 0.75, 1.0, 0.5],
        "baseline_score": [0.4, 0.5, 0.6, 0.8, 0.9, 0.7],
        "improved_score": [0.1, 0.3, 0.5, 0.7, 1.0, 0.5],
    })

    reports = experiment.build_reports(examples)

    assert set(reports) == {"baseline", "improved"}
    for report in reports.values():
        assert set(report.to_dataframe()) == {
            "summary",
            "confusion_matrix",
            "calibration",
            "score_distribution",
            "worst_misses",
            "difficulty_breakdown",
        }


def test_metric_comparison_preserves_metric_direction():
    experiment = _load_experiment_module()
    examples = pd.DataFrame({
        "sample_id": [f"sample-{index}" for index in range(8)],
        "article_id": [f"article-{index}" for index in range(8)],
        "summary": [f"summary-{index}" for index in range(8)],
        "true_label": [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0.75],
        "baseline_score": [0.4, 0.5, 0.6, 0.8, 0.9, 0.7, 0.6, 0.9],
        "improved_score": [0.0, 0.3, 0.5, 0.7, 1.0, 0.5, 0.2, 0.8],
    })

    comparison = experiment.metric_comparison(
        experiment.build_reports(examples)
    )

    assert comparison["metric"].tolist() == [
        "MAE",
        "Spearman correlation",
        "Kendall's tau",
        "Cohen's kappa at 0.5",
        "Brier score",
        "AUC",
    ]
    assert comparison.set_index("metric").loc["MAE", "direction"] == "lower"
    assert (
        comparison.set_index("metric").loc["Spearman correlation", "direction"]
        == "higher"
    )


def test_text_output_mixin_disables_structured_response_schema():
    experiment = _load_experiment_module()

    class StubProvider:
        def _create_chat_completion(
            self, *args, response_format=None, **kwargs
        ):
            return response_format, kwargs

    class TextProvider(experiment._TextOutputCompletionMixin, StubProvider):
        pass

    response_format, kwargs = TextProvider()._create_chat_completion(
        messages=[],
        response_format=object(),
    )
    assert response_format is None
    assert kwargs["seed"] == experiment.MODEL_SEED


def test_figure_module_loads_when_experiment_is_imported():
    experiment = _load_experiment_module()

    plot_publication_figures = experiment._load_publication_figure_function()

    assert callable(plot_publication_figures)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A city council approved a new transit plan.", False),
        ("Police investigated the sexual abuse of a child.", True),
        ("The report described a fatal shooting.", True),
    ],
)
def test_sensitive_news_filter(text, expected):
    experiment = _load_experiment_module()

    assert experiment._contains_sensitive_content(text) is expected

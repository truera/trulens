"""Tests for the benchmark alignment report."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.optional

benchmark = pytest.importorskip("trulens.benchmark")
AlignmentReport = benchmark.AlignmentReport


def _report() -> AlignmentReport:
    return AlignmentReport(
        predicted_scores=[0.1, 0.9, 0.4, 0.8, 0.2, 0.7],
        true_labels=[0.0, 1.0, 1.0, 0.0, 0.2, 0.9],
        examples=[
            {"id": "a", "query": "q0"},
            {"id": "b", "query": "q1"},
            {"id": "c", "query": "q2"},
            {"id": "d", "query": "q3"},
            {"id": "e", "query": "q4"},
            {"id": "f", "query": "q5"},
        ],
        threshold=0.5,
        n_bins=5,
        top_n=3,
    )


def test_construction_accepts_supported_prediction_formats() -> None:
    report = AlignmentReport(
        predicted_scores=np.asarray([0.2, 0.8]),
        true_labels=pd.Series([0.0, 1.0]),
    )
    np.testing.assert_allclose(report.predicted_scores, [0.2, 0.8])

    benchmark_report = AlignmentReport(
        predicted_scores=([0.3, 0.7], [{"meta": 1}, {"meta": 2}]),
        true_labels=[0.0, 1.0],
    )
    np.testing.assert_allclose(benchmark_report.predicted_scores, [0.3, 0.7])

    pair_report = AlignmentReport(
        predicted_scores=[(0.4, 0.9), (0.6, 0.8)],
        true_labels=[0.0, 1.0],
    )
    np.testing.assert_allclose(pair_report.predicted_scores, [0.4, 0.6])


def test_construction_validation_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="equal length"):
        AlignmentReport(predicted_scores=[0.1], true_labels=[0.0, 1.0])

    with pytest.raises(ValueError, match="at least one value"):
        AlignmentReport(predicted_scores=[], true_labels=[])

    with pytest.raises(ValueError, match="finite numeric"):
        AlignmentReport(predicted_scores=[0.1, np.nan], true_labels=[0, 1])

    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        AlignmentReport(predicted_scores=[1.2], true_labels=[1.0])

    with pytest.raises(ValueError, match="thresholds must be a sequence"):
        AlignmentReport(
            predicted_scores=[0.2],
            true_labels=[0.0],
            thresholds=0.5,
        )


def test_summary_metrics_include_required_metric_names() -> None:
    summary = _report().to_dataframe()["summary"]
    metrics = set(summary["metric"])

    assert "MAE" in metrics
    assert "Spearman correlation" in metrics
    assert "Kendall's tau" in metrics
    assert "Brier score" in metrics
    assert "AUC" in metrics
    assert any(metric.startswith("Cohen's kappa") for metric in metrics)


def test_to_dataframe_returns_expected_sections() -> None:
    frames = _report().to_dataframe()

    assert {
        "summary",
        "confusion_matrix",
        "calibration",
        "score_distribution",
        "worst_misses",
        "difficulty_breakdown",
    }.issubset(frames)
    assert all(isinstance(frame, pd.DataFrame) for frame in frames.values())


def test_confusion_matrix_at_default_threshold() -> None:
    confusion = _report().to_dataframe()["confusion_matrix"].iloc[0]

    assert confusion["threshold"] == 0.5
    assert confusion["TN"] == 2
    assert confusion["FP"] == 1
    assert confusion["FN"] == 1
    assert confusion["TP"] == 2


def test_confusion_matrix_supports_custom_thresholds() -> None:
    report = AlignmentReport(
        predicted_scores=[0.2, 0.4, 0.6, 0.8],
        true_labels=[0.1, 0.5, 0.7, 0.9],
        thresholds=[0.3, 0.7],
    )
    confusion = report.to_dataframe()["confusion_matrix"]

    assert confusion["threshold"].tolist() == [0.3, 0.7]
    assert confusion[["TN", "FP", "FN", "TP"]].to_dict("records") == [
        {"TN": 1, "FP": 0, "FN": 0, "TP": 3},
        {"TN": 2, "FP": 0, "FN": 1, "TP": 1},
    ]


def test_worst_misses_are_sorted_by_absolute_error() -> None:
    worst_misses = _report().to_dataframe()["worst_misses"]

    assert worst_misses["index"].tolist() == [3, 2, 5]
    assert worst_misses["absolute_error"].is_monotonic_decreasing
    assert worst_misses.loc[0, "id"] == "d"


def test_calibration_output_has_bins_and_counts() -> None:
    calibration = _report().to_dataframe()["calibration"]

    assert len(calibration) == 5
    assert calibration["count"].sum() == 6
    assert {
        "bin_lower",
        "bin_upper",
        "count",
        "mean_predicted_score",
        "mean_true_label",
    }.issubset(calibration.columns)


def test_difficulty_breakdown_has_expected_buckets() -> None:
    difficulty = _report().to_dataframe()["difficulty_breakdown"]

    assert difficulty["bucket"].tolist() == ["easy", "medium", "hard"]
    assert difficulty["count"].tolist() == [3, 0, 3]


def test_print_summary_returns_and_prints_readable_content(capsys) -> None:
    text = _report().print_summary()
    captured = capsys.readouterr()

    assert "Alignment Report" in text
    assert "Summary metrics" in text
    assert "Alignment Report" in captured.out


def test_to_html_returns_single_html_string() -> None:
    html = _report().to_html()

    assert html.startswith("<html>")
    assert "Alignment Report" in html
    assert "Confusion Matrix" in html
    assert "<table" in html


def test_auc_is_nan_when_labels_are_one_class() -> None:
    report = AlignmentReport(
        predicted_scores=[0.1, 0.2, 0.3],
        true_labels=[0.0, 0.0, 0.0],
    )
    summary = report.to_dataframe()["summary"]
    auc = summary.loc[summary["metric"] == "AUC", "value"].iloc[0]

    assert np.isnan(auc)


def test_plot_returns_matplotlib_figures() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    figure = pytest.importorskip("matplotlib.figure")

    figures = _report().plot()

    assert set(figures) == {"calibration", "score_distribution"}
    assert all(isinstance(value, figure.Figure) for value in figures.values())

    pyplot = pytest.importorskip("matplotlib.pyplot")
    for value in figures.values():
        pyplot.close(value)

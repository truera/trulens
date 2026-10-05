"""Publication figures for the AlignmentReport blog experiment."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def plot_publication_figures(
    reports: dict[str, Any],
    *,
    assets_dir: Path,
    metric_comparison: Callable[[dict[str, Any]], pd.DataFrame],
    n_bins: int,
    threshold: float,
) -> None:
    """Generate the four figures used by the blog post."""

    import matplotlib.pyplot as plt

    assets_dir.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#D97706", "improved": "#2563EB"}
    labels = {"baseline": "Baseline", "improved": "Improved rubric"}

    comparison = metric_comparison(reports)
    positions = np.arange(len(comparison))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        positions - width / 2,
        comparison["baseline"],
        width,
        label=labels["baseline"],
        color=colors["baseline"],
        edgecolor="#1F2937",
        hatch="//",
    )
    axis.bar(
        positions + width / 2,
        comparison["improved"],
        width,
        label=labels["improved"],
        color=colors["improved"],
        edgecolor="#1F2937",
    )
    direction_labels = [
        f"{metric}\n({direction} is better)"
        for metric, direction in zip(
            comparison["metric"],
            comparison["direction"],
            strict=True,
        )
    ]
    axis.set_xticks(positions, direction_labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Metric value")
    axis.set_title("Held-out alignment metrics")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(
        assets_dir / "held_out_metrics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4.8), sharex=True, sharey=True
    )
    for axis, variant in zip(axes, ("baseline", "improved"), strict=True):
        calibration = reports[variant].to_dataframe()["calibration"]
        observed = calibration[calibration["count"] > 0]
        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="#4B5563",
            linestyle="--",
            label="Perfect alignment",
        )
        axis.plot(
            observed["mean_predicted_score"],
            observed["mean_true_label"],
            color=colors[variant],
            marker="o" if variant == "improved" else "s",
            linewidth=2,
            label="Observed",
        )
        axis.set_title(labels[variant])
        axis.set_xlabel("Mean predicted score")
        axis.grid(color="#E5E7EB", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean expert label")
    axes[0].legend(frameon=False)
    figure.suptitle("Held-out calibration by predicted-score bin")
    figure.tight_layout()
    figure.savefig(
        assets_dir / "held_out_calibration.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4.8), sharex=True, sharey=True
    )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for axis, variant in zip(axes, ("baseline", "improved"), strict=True):
        report = reports[variant]
        axis.hist(
            report.true_labels,
            bins=bins,
            color="#E5E7EB",
            edgecolor="#4B5563",
            label="Expert labels",
        )
        axis.hist(
            report.predicted_scores,
            bins=bins,
            histtype="step",
            linewidth=2.5,
            color=colors[variant],
            label=labels[variant],
        )
        axis.set_title(labels[variant])
        axis.set_xlabel("Normalized relevance score")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Examples")
    axes[0].legend(frameon=False)
    figure.suptitle("Held-out score distributions")
    figure.tight_layout()
    figure.savefig(
        assets_dir / "held_out_score_distributions.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    for axis, variant in zip(axes, ("baseline", "improved"), strict=True):
        confusion = reports[variant].to_dataframe()["confusion_matrix"]
        row = confusion.loc[np.isclose(confusion["threshold"], threshold)].iloc[
            0
        ]
        matrix = np.asarray([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        image = axis.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=matrix.max(),
        )
        for row_index in range(2):
            for column_index in range(2):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    int(value),
                    ha="center",
                    va="center",
                    color="white" if image.norm(value) > 0.5 else "#111827",
                    fontsize=13,
                    fontweight="bold",
                )
        axis.set_xticks([0, 1], ["Predicted < 0.5", "Predicted ≥ 0.5"])
        axis.set_yticks([0, 1], ["Expert < 0.5", "Expert ≥ 0.5"])
        axis.set_title(labels[variant])
    figure.suptitle("Held-out confusion matrices at threshold 0.5")
    figure.tight_layout()
    figure.savefig(
        assets_dir / "held_out_confusion_matrices.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

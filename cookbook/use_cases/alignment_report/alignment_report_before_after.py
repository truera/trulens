"""Reproduce the AlignmentReport before-and-after blog experiment.

The experiment evaluates one Gemini judge against expert relevance labels from
SummEval. It uses article-disjoint development, validation, and held-out
splits, caches non-sensitive judge outputs, exports every AlignmentReport
dataframe, and generates the figures used by the blog post.

Run ``python alignment_report_before_after.py --help`` for commands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import numpy as np
import pandas as pd
from trulens.benchmark import AlignmentReport

DATASET_URL = (
    "https://huggingface.co/datasets/mteb/summeval/resolve/"
    "bfc121155064afa2d81b5505682ffc0d96f4334c/data/"
    "test-00000-of-00001-35901af5f6649399.parquet"
)
DATASET_SHA256 = (
    "7ff6f3f4d044223bd6922c5434d27bf28473950346fcfc2a3b7352c19163690e"
)
DATASET_LICENSE = "MIT"
DATASET_SOURCE = "mteb/summeval (derived from Yale-LILY/SummEval)"
MODEL_ID = "gemini-3.1-flash-lite"
TEMPERATURE = 0.0
MODEL_SEED = 123
SELECTION_SEED = 20260731
SAMPLES_PER_BUCKET = 8
THRESHOLD = 0.5
THRESHOLDS = [0.3, 0.5, 0.7]
N_BINS = 5
TOP_N = 5
MAX_ATTEMPTS = 3
REQUEST_INTERVAL_SECONDS = 4.2
SENSITIVE_TOPIC_TERMS = (
    "arrest",
    "babies given antibiotics",
    "behead",
    "blowing people",
    "bombing",
    "child abuse",
    "cerebral palsy",
    "collided",
    "crash",
    "deep cuts",
    "death",
    "dead pet",
    "fatal shooting",
    "homicide",
    "indecent assault",
    "killed",
    "knocking out",
    "miscarriage",
    "migrants",
    "molest",
    "murder",
    "paedophile",
    "pornograph",
    "rape",
    "racist",
    "set the vehicle on fire",
    "sex with",
    "sexual",
    "sexual abuse",
    "shot dead",
    "slave",
    "sniper rifle",
    "stabbing",
    "suicide",
    "terror attack",
    "toddler",
    "two-year-old",
    "underage",
)

BASELINE_CRITERIA = """
Score how relevant the summary is to the source article. A better summary
captures more of the article's important information.
""".strip()

IMPROVED_CRITERIA = """
Judge only content selection: how well the summary captures the source
article's important information. Do not reward fluency, grammaticality,
factual consistency, or length except when they change coverage of important
content.

Build the integer score from three components:
1. Central content, 0-6 points: 0 for no central event or claim; 2 for a vague
   or fragmented reference; 4 for a clear but incomplete account; 6 for a
   clear account of the article's central event or claim.
2. Supporting context, 0-3 points: 0 for none of the important supporting
   context; 1 for a little; 2 for most; 3 for essentially all.
3. Focus, 0-1 point: 1 when the selected content stays focused; 0 when
   repetition or peripheral details materially displace important content.

Add the three components and return that 0-10 total. Repeated versions of the
same fact count once, not as added coverage. A fluent list of names or details
must score low when it does not communicate the article's main point. Do not
require every detail: a concise summary may score highly when it preserves the
lead and the important supporting context.
""".strip()

METRIC_DIRECTIONS = {
    "MAE": "lower",
    "Spearman correlation": "higher",
    "Kendall's tau": "higher",
    f"Cohen's kappa at {THRESHOLD:g}": "higher",
    "Brier score": "lower",
    "AUC": "higher",
}


class _TextOutputCompletionMixin:
    """Work around Gemini rejecting Pydantic's extra-field JSON schema."""

    def _create_chat_completion(
        self,
        *args: Any,
        response_format: Any = None,
        **kwargs: Any,
    ) -> Any:
        del response_format
        request_kwargs = {**kwargs, "seed": MODEL_SEED}
        return super()._create_chat_completion(  # type: ignore[misc]
            *args,
            response_format=None,
            **request_kwargs,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination: Path) -> Path:
    """Download the pinned SummEval parquet if it is not already cached."""

    if destination.exists() and _sha256_file(destination) == DATASET_SHA256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != DATASET_SHA256:
        raise ValueError("Downloaded SummEval file failed the SHA-256 check.")

    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def _score_range(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def _stable_digest(*parts: object) -> str:
    value = ":".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _article_split(article_id: str, seed: int) -> str:
    value = int(_stable_digest(seed, article_id), 16)
    return "development" if value % 2 == 0 else "validation"


def _contains_sensitive_content(text: str) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in SENSITIVE_TOPIC_TERMS)


def _flatten_summeval(source: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row in source.itertuples(index=False):
        article_id = str(source_row.id)
        if _contains_sensitive_content(str(source_row.text)):
            continue
        split = _article_split(article_id, seed)
        for summary_index, (summary, relevance) in enumerate(
            zip(
                source_row.machine_summaries,
                source_row.relevance,
                strict=True,
            )
        ):
            expert_relevance = float(relevance)
            true_label = (expert_relevance - 1.0) / 4.0
            sample_id = f"{article_id}:M{summary_index}"
            rows.append({
                "sample_id": sample_id,
                "article_id": article_id,
                "summary_index": summary_index,
                "source": str(source_row.text),
                "summary": str(summary),
                "expert_relevance": expert_relevance,
                "true_label": true_label,
                "score_range": _score_range(true_label),
                "split": split,
                "selection_key": _stable_digest(seed, sample_id),
            })
    return pd.DataFrame(rows)


def select_examples(
    source: pd.DataFrame,
    *,
    samples_per_bucket: int = SAMPLES_PER_BUCKET,
    seed: int = SELECTION_SEED,
) -> pd.DataFrame:
    """Select balanced, deterministic, article-disjoint examples."""

    flattened = _flatten_summeval(source, seed)
    selected_groups = []
    for split in ("development", "validation"):
        for score_range in ("low", "medium", "high"):
            candidates = flattened[
                (flattened["split"] == split)
                & (flattened["score_range"] == score_range)
            ].sort_values("selection_key", kind="mergesort")
            if len(candidates) < samples_per_bucket:
                raise ValueError(
                    f"Not enough {split}/{score_range} examples: "
                    f"need {samples_per_bucket}, found {len(candidates)}."
                )
            selected_groups.append(candidates.head(samples_per_bucket))

    selected_article_ids = set(
        pd.concat(selected_groups, ignore_index=True)["article_id"]
    )
    for score_range in ("low", "medium", "high"):
        candidates = flattened[
            (~flattened["article_id"].isin(selected_article_ids))
            & (flattened["score_range"] == score_range)
        ].sort_values("selection_key", kind="mergesort")
        if len(candidates) < samples_per_bucket:
            raise ValueError(
                f"Not enough held_out/{score_range} examples: "
                f"need {samples_per_bucket}, found {len(candidates)}."
            )
        selected_groups.append(
            candidates.head(samples_per_bucket).assign(split="held_out")
        )

    return (
        pd
        .concat(selected_groups, ignore_index=True)
        .sort_values(["split", "score_range", "selection_key"])
        .drop(columns=["selection_key"])
        .reset_index(drop=True)
    )


def _prompt_hash(criteria: str) -> str:
    return _stable_digest(
        MODEL_ID,
        TEMPERATURE,
        MODEL_SEED,
        criteria,
        "min_score=0",
        "max_score=10",
    )[:16]


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected an object in cache file {path}.")
    return data


def _write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _score_result(result: Any) -> tuple[float, Any]:
    if isinstance(result, tuple):
        score, metadata = result
        return float(score), metadata
    return float(result), {}


def _score_example(
    provider: Any,
    *,
    source: str,
    summary: str,
    criteria: str,
) -> tuple[float, Any]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = provider.relevance_with_cot_reasons(
                prompt=source,
                response=summary,
                criteria=criteria,
                min_score_val=0,
                max_score_val=10,
                temperature=TEMPERATURE,
            )
            score, metadata = _score_result(result)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Judge returned invalid score {score!r}.")
            return score, metadata
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("Retry loop exited unexpectedly.")


def score_variant(
    examples: pd.DataFrame,
    *,
    variant: str,
    criteria: str,
    cache_path: Path,
) -> pd.DataFrame:
    """Score uncached rows for one judge variant."""

    from trulens.providers.google import provider as google_provider

    class TextOutputGoogleProvider(
        _TextOutputCompletionMixin,
        google_provider.Google,
    ):
        pass

    provider = TextOutputGoogleProvider(model_engine=MODEL_ID)
    cache = _load_cache(cache_path)
    prompt_hash = _prompt_hash(criteria)
    scores: list[float] = []
    for row_number, row in enumerate(examples.itertuples(index=False), start=1):
        cache_key = f"{variant}:{prompt_hash}:{row.sample_id}"
        cached = cache.get(cache_key)
        if cached is None:
            print(
                f"[{variant}] {row_number}/{len(examples)} {row.sample_id}",
                file=sys.stderr,
            )
            score, metadata = _score_example(
                provider,
                source=row.source,
                summary=row.summary,
                criteria=criteria,
            )
            cached = {
                "sample_id": row.sample_id,
                "variant": variant,
                "model": MODEL_ID,
                "temperature": TEMPERATURE,
                "model_seed": MODEL_SEED,
                "prompt_hash": prompt_hash,
                "score": score,
                "metadata": metadata,
                "run_date": dt.datetime.now(dt.UTC).date().isoformat(),
            }
            cache = {**cache, cache_key: cached}
            _write_cache(cache_path, cache)
            time.sleep(REQUEST_INTERVAL_SECONDS)
        scores.append(float(cached["score"]))

    return examples.assign(**{f"{variant}_score": scores})


def _cached_scores(
    examples: pd.DataFrame,
    *,
    variant: str,
    criteria: str,
    cache_path: Path,
) -> list[float]:
    cache = _load_cache(cache_path)
    prompt_hash = _prompt_hash(criteria)
    scores = []
    for sample_id in examples["sample_id"]:
        cache_key = f"{variant}:{prompt_hash}:{sample_id}"
        if cache_key not in cache:
            raise ValueError(
                f"Missing cached {variant} score for {sample_id}. "
                "Run the corresponding scoring stage first."
            )
        scores.append(float(cache[cache_key]["score"]))
    return scores


def build_reports(examples: pd.DataFrame) -> dict[str, AlignmentReport]:
    """Construct baseline and improved reports using the current public API."""

    report_examples = examples[
        ["sample_id", "article_id", "summary"]
    ].reset_index(drop=True)
    common = {
        "true_labels": examples["true_label"].tolist(),
        "examples": report_examples,
        "threshold": THRESHOLD,
        "thresholds": THRESHOLDS,
        "n_bins": N_BINS,
        "top_n": TOP_N,
    }
    return {
        "baseline": AlignmentReport(
            predicted_scores=examples["baseline_score"].tolist(),
            **common,
        ),
        "improved": AlignmentReport(
            predicted_scores=examples["improved_score"].tolist(),
            **common,
        ),
    }


def metric_comparison(
    reports: dict[str, AlignmentReport],
) -> pd.DataFrame:
    """Return aligned summary metrics and their preferred direction."""

    baseline = (
        reports["baseline"]
        .to_dataframe()["summary"]
        .rename(columns={"value": "baseline"})
    )
    improved = (
        reports["improved"]
        .to_dataframe()["summary"]
        .rename(columns={"value": "improved"})
    )
    comparison = baseline.merge(improved, on="metric", validate="one_to_one")
    return comparison.assign(
        direction=comparison["metric"].map(METRIC_DIRECTIONS)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def _export_report_frames(
    reports: dict[str, AlignmentReport],
    *,
    split: str,
    output_dir: Path,
) -> None:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for variant, report in reports.items():
        for section, frame in report.to_dataframe().items():
            frame.to_csv(
                frames_dir / f"{split}_{variant}_{section}.csv",
                index=False,
            )
        (output_dir / f"{split}_{variant}_report.html").write_text(
            report.to_html(),
            encoding="utf-8",
        )
        for plot_name, figure in report.plot().items():
            figure.savefig(
                output_dir / f"{split}_{variant}_{plot_name}.png",
                dpi=160,
                bbox_inches="tight",
            )


def export_results(
    selected: pd.DataFrame,
    *,
    cache_path: Path,
    output_dir: Path,
    assets_dir: Path,
) -> None:
    """Load cached scores and export reports, dataframes, and figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_splits: dict[str, pd.DataFrame] = {}
    result_payload: dict[str, Any] = {
        "run_metadata": {
            "provider": "Google Gemini Developer API",
            "model": MODEL_ID,
            "temperature": TEMPERATURE,
            "model_seed": MODEL_SEED,
            "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
            "selection_seed": SELECTION_SEED,
            "run_date": dt.datetime.now(dt.UTC).date().isoformat(),
            "threshold": THRESHOLD,
            "thresholds": THRESHOLDS,
            "n_bins": N_BINS,
            "top_n": TOP_N,
        },
        "dataset": {
            "source": DATASET_SOURCE,
            "url": DATASET_URL,
            "sha256": DATASET_SHA256,
            "license": DATASET_LICENSE,
            "original_examples": 1600,
            "selected_examples": len(selected),
            "label_mapping": "(expert relevance mean - 1) / 4",
            "sensitive_topic_filter": list(SENSITIVE_TOPIC_TERMS),
            "selection_counts": (
                selected
                .groupby(["split", "score_range"])
                .size()
                .rename("count")
                .reset_index()
                .to_dict("records")
            ),
        },
        "rubrics": {
            "baseline": BASELINE_CRITERIA,
            "improved": IMPROVED_CRITERIA,
        },
        "splits": {},
    }
    for split in ("development", "validation", "held_out"):
        split_examples = selected[selected["split"] == split].reset_index(
            drop=True
        )
        scored = split_examples.assign(
            baseline_score=_cached_scores(
                split_examples,
                variant="baseline",
                criteria=BASELINE_CRITERIA,
                cache_path=cache_path,
            ),
            improved_score=_cached_scores(
                split_examples,
                variant="improved",
                criteria=IMPROVED_CRITERIA,
                cache_path=cache_path,
            ),
        )
        reports = build_reports(scored)
        _export_report_frames(
            reports,
            split=split,
            output_dir=output_dir,
        )
        scored_splits[split] = scored
        result_payload["splits"][split] = {
            "count": len(scored),
            "metrics": metric_comparison(reports).to_dict("records"),
            "confusion_matrices": {
                variant: report.to_dataframe()["confusion_matrix"].to_dict(
                    "records"
                )
                for variant, report in reports.items()
            },
            "difficulty_breakdown": {
                variant: report.to_dataframe()["difficulty_breakdown"].to_dict(
                    "records"
                )
                for variant, report in reports.items()
            },
            "worst_misses": {
                variant: report.to_dataframe()["worst_misses"].to_dict(
                    "records"
                )
                for variant, report in reports.items()
            },
        }

    held_out_reports = build_reports(scored_splits["held_out"])
    _load_publication_figure_function()(
        held_out_reports,
        assets_dir=assets_dir,
        metric_comparison=metric_comparison,
        n_bins=N_BINS,
        threshold=THRESHOLD,
    )
    score_columns = [
        "sample_id",
        "article_id",
        "summary_index",
        "split",
        "score_range",
        "expert_relevance",
        "true_label",
        "baseline_score",
        "improved_score",
    ]
    pd.concat(scored_splits.values(), ignore_index=True)[score_columns].to_csv(
        output_dir / "scores.csv",
        index=False,
    )
    (output_dir / "alignment_report_results.json").write_text(
        json.dumps(_json_safe(result_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (assets_dir / "alignment_report_results.json").write_text(
        json.dumps(_json_safe(result_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_publication_figure_function():
    """Load the adjacent figure module in CLI and import-based executions."""

    import importlib.util

    module_path = Path(__file__).with_name("alignment_report_figures.py")
    spec = importlib.util.spec_from_file_location(
        "_alignment_report_figures",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load figure module at {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_publication_figures


def _stage_variants(stage: str) -> tuple[tuple[str, str, str], ...]:
    stages = {
        "baseline-development": (
            ("development", "baseline", BASELINE_CRITERIA),
        ),
        "improved-development": (
            ("development", "improved", IMPROVED_CRITERIA),
        ),
        "validation": (
            ("validation", "baseline", BASELINE_CRITERIA),
            ("validation", "improved", IMPROVED_CRITERIA),
        ),
        "held-out": (
            ("held_out", "baseline", BASELINE_CRITERIA),
            ("held_out", "improved", IMPROVED_CRITERIA),
        ),
        "all": (
            ("development", "baseline", BASELINE_CRITERIA),
            ("development", "improved", IMPROVED_CRITERIA),
            ("validation", "baseline", BASELINE_CRITERIA),
            ("validation", "improved", IMPROVED_CRITERIA),
            ("held_out", "baseline", BASELINE_CRITERIA),
            ("held_out", "improved", IMPROVED_CRITERIA),
        ),
    }
    return stages.get(stage, ())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "baseline-development",
            "improved-development",
            "validation",
            "held-out",
            "all",
            "report",
        ],
        default="prepare",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "trulens" / "alignment_report_blog",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(__file__).parents[4]
        / "docs"
        / "blog"
        / "assets"
        / "alignment_report_before_after",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.cache_dir / "exports"

    dataset_path = download_dataset(args.cache_dir / "summeval.parquet")
    source = pd.read_parquet(dataset_path)
    selected = select_examples(source)
    selection_path = args.cache_dir / "selected_examples.csv"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selection_path, index=False)
    print(
        selected.groupby(["split", "score_range"]).size().to_string(),
        file=sys.stderr,
    )

    if args.stage == "prepare":
        print(f"Prepared {len(selected)} examples at {selection_path}")
        return

    cache_path = args.cache_dir / "judge_outputs.json"
    if args.stage == "report":
        export_results(
            selected,
            cache_path=cache_path,
            output_dir=output_dir,
            assets_dir=args.assets_dir,
        )
        return

    scored = selected
    for split, variant, criteria in _stage_variants(args.stage):
        split_examples = scored[scored["split"] == split].reset_index(drop=True)
        score_variant(
            split_examples,
            variant=variant,
            criteria=criteria,
            cache_path=cache_path,
        )
    print(f"Cached judge outputs at {cache_path}")


if __name__ == "__main__":
    main()

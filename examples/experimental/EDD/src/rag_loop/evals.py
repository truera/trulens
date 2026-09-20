from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import numpy as np

from trulens.apps.langgraph import TruGraph
from trulens.core import Metric, Selector, TruSession
from trulens.providers.openai import OpenAI

from .config import Settings, get_settings
from .dataset import load_validated_items
from .rag import build_graph

APP_NAME = "aircraft-systems"
DATASET_PATH = Path("evals/seed.jsonl")


def get_feedback_functions(settings: Settings) -> list[Metric]:
    """Configure TruLens RAG Triad feedback functions."""
    provider = OpenAI(
        model_engine=settings.eval_model,
        api_key=settings.openai_api_key or None,
    )

    f_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons_consider_answerability,
        name="groundedness",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
            "question": Selector.select_record_input(),
        },
    )

    f_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="answer_relevance",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    f_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="context_relevance",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )

    return [f_groundedness, f_answer_relevance, f_context_relevance]


@dataclass
class EvaluationResult:
    experiment_name: str
    dataset_items: list[dict[str, Any]]
    records_df: Any
    feedback_cols: list[str]
    aggregated_scores: dict[str, dict[str, float]]
    experiment_scores: dict[str, dict[str, Any]]
    cases: list[dict[str, Any]]


def run_evaluation(
    settings: Settings | None = None,
    experiment_name: str | None = None,
    dataset_path: Path = DATASET_PATH,
) -> EvaluationResult:
    settings = settings or get_settings()
    exp_name = experiment_name or f"rag-{settings.chat_model}-{settings.retrieval_k}"

    session = TruSession(database_url=settings.database_url)
    items = load_validated_items(dataset_path)
    feedbacks = get_feedback_functions(settings)

    graph = build_graph(settings)
    tru_graph = TruGraph(
        graph,
        app_name=APP_NAME,
        app_version=exp_name,
        feedbacks=feedbacks,
    )

    outputs = []
    with tru_graph as recording:
        for item in items:
            res = tru_graph.app.invoke({"question": item["input"]})
            outputs.append(res.get("answer", ""))

    session.force_flush()
    records_df = recording.retrieve_feedback_results(timeout=180)

    # Extract feedback column names
    feedback_cols = [f.name for f in feedbacks]

    # Compute aggregated scores
    aggregated_scores: dict[str, dict[str, float]] = {}
    for col in feedback_cols:
        if col in records_df.columns:
            vals = records_df[col].dropna().tolist()
            if vals:
                aggregated_scores[col] = {
                    "mean": float(np.mean(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                }
            else:
                aggregated_scores[col] = {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0,
                }

    # Build per-case records
    cases = []
    for idx, item in enumerate(items):
        case_scores: dict[str, dict[str, Any]] = {}
        for col in feedback_cols:
            score_val = None
            reason_val = ""
            if col in records_df.columns and idx < len(records_df):
                val = records_df[col].iloc[idx]
                score_val = float(val) if not np.isnan(val) else None
            calls_col = f"{col}_calls"
            if calls_col in records_df.columns and idx < len(records_df):
                calls_data = records_df[calls_col].iloc[idx]
                if isinstance(calls_data, list) and calls_data:
                    meta = calls_data[0].get("meta", {})
                    reason_val = (
                        meta.get("reason") or meta.get("reasons") or ""
                    )

            case_scores[col] = {
                "value": score_val,
                "reason": str(reason_val),
                "scoring_failed": score_val is None,
            }

        cases.append({
            "dataset_item_id": idx + 1,
            "input": item["input"],
            "expected_output": item["expected_output"],
            "label": item.get("label", 1),
            "output": outputs[idx] if idx < len(outputs) else "",
            "scores": case_scores,
        })

    return EvaluationResult(
        experiment_name=exp_name,
        dataset_items=items,
        records_df=records_df,
        feedback_cols=feedback_cols,
        aggregated_scores=aggregated_scores,
        experiment_scores={},
        cases=cases,
    )


def run_config_loop(
    configurations: list[dict[str, Any]], settings: Settings | None = None
) -> list[EvaluationResult]:
    settings = settings or get_settings()
    results = []
    for configuration in configurations:
        experiment_settings = replace(
            settings,
            chat_model=configuration.get("chat_model", settings.chat_model),
            retrieval_k=configuration.get("retrieval_k", settings.retrieval_k),
        )
        results.append(
            run_evaluation(
                experiment_settings,
                experiment_name=configuration["name"],
            )
        )
    return results

from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
from typing import Any

from .evals import EvaluationResult

RESULTS_DIR = Path("evals/results")


def to_result_record(experiment: dict[str, Any]) -> dict[str, Any]:
    evaluation: EvaluationResult = experiment["evaluation"]
    return {
        "name": experiment["name"],
        "change": experiment.get("change", ""),
        "acceptance": experiment.get("acceptance", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregated_scores": evaluation.aggregated_scores,
        "experiment_scores": evaluation.experiment_scores,
        "cases": evaluation.cases,
    }


def save_result(
    record: dict[str, Any], results_dir: Path = RESULTS_DIR
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record['name']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def load_result(name: str, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    path = results_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No persisted result for experiment {name!r} at {path}. "
            f"Run it first with `rag-loop run {name}`."
        )
    return json.loads(path.read_text())

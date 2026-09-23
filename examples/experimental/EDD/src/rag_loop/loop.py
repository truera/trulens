from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Any

from trulens.core import TruSession

from .config import Settings
from .config import get_settings
from .evals import run_evaluation
from .report import Verdict
from .report import decide
from .results import save_result
from .results import to_result_record

SOURCE_PATHS = ["src"]
RECORD_PATHS = ["evals/experiments.json", "evals/results"]


def load_experiments(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def run_experiment(
    experiment: dict[str, Any],
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    experiment_settings = replace(
        settings,
        chat_model=experiment.get("chat_model", settings.chat_model),
        retrieval_k=experiment.get("retrieval_k", settings.retrieval_k),
    )
    evaluation = run_evaluation(
        experiment_settings,
        experiment_name=experiment["name"],
    )
    return {
        "name": experiment["name"],
        "change": experiment.get("change", ""),
        "acceptance": experiment.get("acceptance", {}),
        "evaluation": evaluation,
    }


def run_experiment_loop(
    experiments_path: Path = Path("evals/experiments.json"),
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    return [
        run_experiment(experiment, settings)
        for experiment in load_experiments(experiments_path)
    ]


def _record_decision_in_dashboard(
    app_id: str, verdict: Verdict, settings: Settings
) -> None:
    """Persist the keep/reject verdict onto the app so the TruLens dashboard
    leaderboard shows an 'accepted/rejected' column for each experiment.

    Uses the same database as the evaluation run so the dashboard (which must
    point at the same TRULENS_DATABASE_URL) can read it. update_app_metadata
    merges, so the config metadata set at app-creation time is preserved.
    """
    accepted = verdict.decision in ("keep", "no_baseline")
    try:
        session = TruSession(database_url=settings.database_url)
        session.connector.db.update_app_metadata(
            app_id,
            {
                "decision": verdict.decision,
                "accepted": accepted,
                "verdict_reason": verdict.reason,
            },
        )
    except Exception as exc:  # noqa: BLE001 - dashboard metadata is best-effort
        print(f"[warn] Could not record decision in dashboard metadata: {exc}")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=check
    )


def _commit(paths: list[str], message: str) -> None:
    _git("add", *paths)
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    _git("commit", "-m", message)


def _revert(paths: list[str]) -> None:
    _git("restore", "--worktree", "--staged", *paths)


def run_iteration(
    name: str,
    experiments_path: Path = Path("evals/experiments.json"),
    settings: Settings | None = None,
) -> Verdict:
    settings = settings or get_settings()
    experiments = load_experiments(experiments_path)
    entry = next((e for e in experiments if e["name"] == name), None)
    if entry is None:
        raise ValueError(f"No entry named {name!r} in {experiments_path}")

    result = run_experiment(entry, settings)
    save_result(to_result_record(result))

    verdict = decide(name, experiments)
    _record_decision_in_dashboard(
        result["evaluation"].app_id, verdict, settings
    )
    entry["result"] = {
        "decision": verdict.decision,
        "reason": verdict.reason,
        "plateaued": verdict.plateaued,
        "flat": bool(verdict.comparisons)
        and all(
            c.delta is not None and c.delta == 0 for c in verdict.comparisons
        ),
        "comparisons": [asdict(c) for c in verdict.comparisons],
    }
    experiments_path.write_text(json.dumps(experiments, indent=2) + "\n")

    if verdict.decision == "reject":
        _revert(SOURCE_PATHS)
        _commit(RECORD_PATHS, f"{name} (rejected): {entry.get('change', '')}")
    else:
        _commit(
            SOURCE_PATHS + RECORD_PATHS,
            f"{name}: {entry.get('change', '')}",
        )

    return verdict

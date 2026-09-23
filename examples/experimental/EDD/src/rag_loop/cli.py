import argparse
import json
import logging
from pathlib import Path
import warnings

from trulens.core import TruSession

# Suppress UserWarnings from feedback providers and instrumentation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="trulens")

# Silence the optional-package "not present in requirements" warnings emitted
# during instrumentation (e.g. jsonschema).
logging.getLogger("trulens.core.utils.imports").setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph RAG app with TruLens evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Index the source PDF into Qdrant")
    subparsers.add_parser("seed", help="Validate the golden evaluation dataset")
    subparsers.add_parser(
        "evaluate", help="Run TruLens evaluation on the baseline"
    )
    loop_parser = subparsers.add_parser(
        "loop", help="Run configured multi-config sweep"
    )
    loop_parser.add_argument("--config", default="evals/configs.json")
    experiment_parser = subparsers.add_parser(
        "experiment-loop", help="Run all experiments in experiments.json"
    )
    experiment_parser.add_argument("--config", default="evals/experiments.json")
    run_parser = subparsers.add_parser(
        "run",
        help="Run one experiment, persist it, and decide keep/reject/revert",
    )
    run_parser.add_argument("name")
    run_parser.add_argument("--config", default="evals/experiments.json")
    report_parser = subparsers.add_parser(
        "report", help="Show the worst-scoring cases for an experiment"
    )
    report_parser.add_argument("name")
    report_parser.add_argument("--worst", type=int, default=5)
    report_parser.add_argument("--metric", default=None)
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question against the indexed PDF"
    )
    ask_parser.add_argument("question")
    subparsers.add_parser("dashboard", help="Launch the TruLens dashboard UI")
    args = parser.parse_args()

    try:
        if args.command == "index":
            from .indexing import index_pdf

            print(f"Indexed {index_pdf()} chunks.")
        elif args.command == "seed":
            from .dataset import load_validated_items

            items = load_validated_items(Path("evals/seed.jsonl"))
            print(f"Dataset validated successfully: {len(items)} golden items.")
        elif args.command == "evaluate":
            from .evals import run_evaluation

            result = run_evaluation()
            print(f"Evaluation complete for {result.experiment_name}.")
            for metric, stats in result.aggregated_scores.items():
                print(f"  {metric}: mean={stats['mean']:.3f}")
        elif args.command == "loop":
            from .evals import run_config_loop

            configurations = json.loads(Path(args.config).read_text())
            for result in run_config_loop(configurations):
                print(f"Result for {result.experiment_name}:")
                for metric, stats in result.aggregated_scores.items():
                    print(f"  {metric}: mean={stats['mean']:.3f}")
        elif args.command == "experiment-loop":
            from .loop import run_experiment_loop

            for result in run_experiment_loop(Path(args.config)):
                print(result)
        elif args.command == "run":
            from .loop import run_iteration

            verdict = run_iteration(args.name, Path(args.config))
            print(f"{verdict.decision}: {verdict.reason}")
            for comparison in verdict.comparisons:
                print(
                    f"  {comparison.metric}: {comparison.baseline} -> {comparison.current} (delta {comparison.delta})"
                )
            if verdict.plateaued:
                print(
                    "Plateaued: no improvement over the last few kept runs. Consider a structural change."
                )
        elif args.command == "report":
            from .report import load_and_format

            print(load_and_format(args.name, n=args.worst, metric=args.metric))
        elif args.command == "dashboard":
            try:
                from trulens.dashboard import run_dashboard
            except ImportError:
                print(
                    "trulens-dashboard is not installed. Please install it with:"
                )
                print("  pip install trulens-dashboard")
                return

            session = TruSession()
            print("Starting TruLens dashboard...")
            proc = run_dashboard(session)
            try:
                proc.wait()
            except KeyboardInterrupt:
                pass
        else:
            from .rag import answer_question

            result = answer_question(args.question)
            print(result["answer"])
    finally:
        TruSession().force_flush()


if __name__ == "__main__":
    main()

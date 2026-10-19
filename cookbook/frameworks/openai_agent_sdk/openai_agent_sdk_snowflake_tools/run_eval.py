"""Batch evaluation using Snowflake AI Observability Runs."""

import datetime
import logging
import time

import pandas as pd

from trulens.core.run import RunConfig, RunStatus

from src.eval.ground_truth import (
    TEST_QUERIES_ANALYST,
    TEST_QUERIES_MIXED,
    TEST_QUERIES_SEARCH,
)
from src.eval.metrics import SERVERSIDE_METRICS
from src.observability.trulens_setup import setup_observability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

POLL_INTERVAL = 15
MAX_POLL_WAIT = 300


def wait_for_invocation(run, run_name, timeout=MAX_POLL_WAIT):
    terminal = {
        RunStatus.INVOCATION_COMPLETED,
        RunStatus.INVOCATION_PARTIALLY_COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }
    elapsed = 0
    while elapsed < timeout:
        status = run.get_status()
        logger.info(f"[{run_name}] status: {status} (waited {elapsed}s)")
        if status in terminal:
            return status
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    return run.get_status()


def main():
    agent_app, tru_app, session, sf_connector, custom_metrics = (
        setup_observability()
    )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"BATCH_EVAL_RUN_{ts}"

    queries = (
        TEST_QUERIES_ANALYST + TEST_QUERIES_SEARCH + TEST_QUERIES_MIXED
    )
    input_df = pd.DataFrame({"question": queries})

    print(f"\n{'=' * 60}")
    print(f"Creating run '{run_name}' with {len(queries)} queries")
    print(f"{'=' * 60}\n")

    run_config = RunConfig(
        run_name=run_name,
        dataset_name=f"{run_name}_dataset",
        source_type="DATAFRAME",
        dataset_spec={"input": "question"},
        description="Batch evaluation of Support Intelligence Agent",
        llm_judge_name="openai-gpt-5.1",
    )

    run = tru_app.add_run(run_config=run_config)

    print("Starting run (invoking agent on all queries)...")
    run.start(input_df=input_df)
    print("Invocation complete. Waiting for span ingestion...\n")

    status = wait_for_invocation(run, run_name)
    print(f"Invocation status: {status}\n")

    if status not in {
        RunStatus.INVOCATION_COMPLETED,
        RunStatus.INVOCATION_PARTIALLY_COMPLETED,
    }:
        print(f"Invocation did not succeed ({status}). Skipping metrics.")
        return

    all_metrics = custom_metrics + SERVERSIDE_METRICS
    print(
        f"Computing metrics ({len(custom_metrics)} client-side"
        f" + {len(SERVERSIDE_METRICS)} server-side)..."
    )
    try:
        result = run.compute_metrics(metrics=all_metrics)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\n{'=' * 60}")
    print("Done. Check Snowsight Evaluations UI for results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

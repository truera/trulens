"""Benchmark: compare old (fetch-all) vs new (SQL aggregate) leaderboard paths.

Usage:
    TRULENS_OTEL_TRACING=1 python scripts/benchmark_dashboard.py [--db test_perf.sqlite] [--runs 5]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ["TRULENS_OTEL_TRACING"] = "1"

from trulens.core.database.sqlalchemy import SQLAlchemyDB


def bench(fn, runs: int = 5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark leaderboard query paths"
    )
    parser.add_argument(
        "--db", default="test_perf.sqlite", help="SQLite DB path"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of benchmark runs"
    )
    args = parser.parse_args()

    db_url = f"sqlite:///{args.db}"
    db = SQLAlchemyDB.from_db_url(db_url)
    db.migrate_database()

    print(f"Benchmarking with {args.db} ({args.runs} runs each)\n")
    print(f"{'Operation':<45} {'Old (s)':<12} {'New (s)':<12} {'Speedup':<10}")
    print("-" * 79)

    results = {}

    old_times, (_old_df, _old_cols) = bench(
        lambda: db.get_records_and_feedback(app_name="rag_chatbot"),
        runs=args.runs,
    )
    new_times, (_new_df, _new_cols) = bench(
        lambda: db.get_leaderboard_aggregates(app_name="rag_chatbot"),
        runs=args.runs,
    )
    _report(
        "Leaderboard (single app, 3 versions)",
        old_times,
        new_times,
        results,
    )

    old_times, _ = bench(
        lambda: db.get_records_and_feedback(app_name="rag_chatbot", limit=1000),
        runs=args.runs,
    )
    new_times, _ = bench(
        lambda: db.get_leaderboard_aggregates(app_name="rag_chatbot"),
        runs=args.runs,
    )
    _report(
        "Leaderboard (single app, limit=1000)",
        old_times,
        new_times,
        results,
    )

    old_times, _ = bench(
        lambda: db.get_records_and_feedback(),
        runs=args.runs,
    )
    new_times, _ = bench(
        lambda: db.get_leaderboard_aggregates(),
        runs=args.runs,
    )
    _report(
        "Leaderboard (all apps, 15 versions)",
        old_times,
        new_times,
        results,
    )

    old_times, _ = bench(
        lambda: db.get_records_and_feedback(
            app_name="rag_chatbot",
            app_versions=["v1.0"],
        ),
        runs=args.runs,
    )
    new_times, _ = bench(
        lambda: db.get_leaderboard_aggregates(
            app_name="rag_chatbot",
            app_versions=["v1.0"],
        ),
        runs=args.runs,
    )
    _report(
        "Leaderboard (single version)",
        old_times,
        new_times,
        results,
    )

    print("\n## Aggregated Results")
    print(
        f"\n{'Operation':<45} {'Old mean':<12} {'New mean':<12} {'Speedup':<10}"
    )
    print("-" * 79)
    for op, data in results.items():
        print(
            f"{op:<45} {data['old_mean']:<12.3f} {data['new_mean']:<12.3f} {data['speedup']:<10.1f}x"
        )


def _report(label, old_times, new_times, results):
    old_mean = statistics.mean(old_times)
    new_mean = statistics.mean(new_times)
    speedup = old_mean / new_mean if new_mean > 0 else float("inf")
    old_std = statistics.stdev(old_times) if len(old_times) > 1 else 0
    new_std = statistics.stdev(new_times) if len(new_times) > 1 else 0
    print(
        f"{label:<45} {old_mean:.3f}±{old_std:.3f}  {new_mean:.3f}±{new_std:.3f}  {speedup:.1f}x"
    )
    results[label] = {
        "old_mean": old_mean,
        "old_std": old_std,
        "new_mean": new_mean,
        "new_std": new_std,
        "speedup": speedup,
    }


if __name__ == "__main__":
    main()

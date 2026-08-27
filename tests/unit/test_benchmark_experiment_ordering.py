"""The benchmark experiment must return scores in ground-truth row order.

`TruBenchmarkExperiment.__call__` says so in its own docstring: "Note the order
of generated scores must be preserved to match the order of the true labels."
`GroundTruthAggregator` then zips those scores against `true_labels`
positionally, so any drop or reorder is scored against the wrong row and
nothing raises.

The reassembly used to key futures by the frame's index LABEL, taken from
`iterrows()`, and then read them back over `range(len(ground_truth))`. That is
only equivalent when the labels happen to be 0..n-1. A frame left behind by a
filter, a `set_index` or a `concat` breaks it three different ways, all silent.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import pandas as pd
import pytest


def _reassemble(ground_truth: pd.DataFrame) -> list[str]:
    """The reassembly shape used by TruBenchmarkExperiment.__call__.

    The score is the row's own query, so a wrong order is visible in the output
    rather than hidden behind a float.
    """
    scores: list[str] = []
    future_to_index: dict = {}
    index_to_results: dict = {}
    with ThreadPoolExecutor() as executor:
        for position, (_, row) in enumerate(ground_truth.iterrows()):
            future = executor.submit(lambda q: q, row["query"])
            future_to_index[future] = position
        for future in as_completed(future_to_index):
            position = future_to_index[future]
            index_to_results.setdefault(position, []).append(future.result())
        for position in range(len(ground_truth)):
            if position in index_to_results:
                for ret in index_to_results[position]:
                    scores.append(ret)
    return scores


ROWS = [{"query": f"q{i}"} for i in range(4)]


@pytest.mark.parametrize(
    ("name", "frame"),
    [
        ("default range index", pd.DataFrame(ROWS)),
        (
            "left by a filter",
            pd.DataFrame(ROWS * 2).query("query != 'q0'").head(4),
        ),
        ("string index", pd.DataFrame(ROWS).set_index(pd.Index(list("abcd")))),
        (
            "left by a concat, duplicate labels",
            pd.concat([pd.DataFrame(ROWS[:2]), pd.DataFrame(ROWS[2:])]),
        ),
    ],
)
def test_scores_follow_ground_truth_row_order(name: str, frame: pd.DataFrame):
    """Every row is scored exactly once, in the order the frame holds them.

    On the previous reassembly: the filtered frame returned 3 scores for 4
    rows, the string-indexed frame returned an empty list, and the concatenated
    frame returned 4 scores in the wrong order because the duplicate labels
    were emitted in completion order.
    """
    expected = list(frame["query"])
    assert _reassemble(frame) == expected, name

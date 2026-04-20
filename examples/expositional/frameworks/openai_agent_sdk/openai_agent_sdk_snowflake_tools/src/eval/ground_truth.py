"""Ground truth data and test query suites.

Ground truth is loaded from Snowflake tables:
- GROUND_TRUTH_SEARCH: expected retrieval chunks per query
- GROUND_TRUTH_ANALYST: golden SQL per query

Test queries are loaded from test_queries.json. To add more test cases,
edit that file directly — no code changes needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from snowflake.snowpark import Session

_QUERIES_PATH = Path(__file__).parent / "test_queries.json"


def load_test_queries() -> dict[str, list[str]]:
    """Load test queries from test_queries.json.

    Returns dict with keys 'analyst', 'search', 'mixed'.
    """
    with open(_QUERIES_PATH) as f:
        return json.load(f)


def load_search_ground_truth(session: Session) -> list[dict]:
    """Load search ground truth from Snowflake, grouped by query.

    Returns list of dicts matching GroundTruthAgreement format:
        [{"query": "...", "expected_chunks": [{"text": "...", "expect_score": 1.0}]}]
    """
    rows = session.sql(
        "SELECT QUERY, EXPECTED_CHUNK_TEXT, EXPECT_SCORE "
        "FROM SUPPORT_INTELLIGENCE.DATA.GROUND_TRUTH_SEARCH"
    ).collect()
    grouped: dict[str, dict] = {}
    for row in rows:
        q = row["QUERY"]
        if q not in grouped:
            grouped[q] = {"query": q, "expected_chunks": []}
        grouped[q]["expected_chunks"].append({
            "text": row["EXPECTED_CHUNK_TEXT"],
            "expect_score": float(row["EXPECT_SCORE"]),
        })
    return list(grouped.values())


def load_analyst_golden_sql(session: Session) -> dict[str, str]:
    """Load analyst golden SQL ground truth from Snowflake.

    Returns dict mapping query -> golden SQL string.
    """
    rows = session.sql(
        "SELECT QUERY, GOLDEN_SQL "
        "FROM SUPPORT_INTELLIGENCE.DATA.GROUND_TRUTH_ANALYST"
    ).collect()
    return {row["QUERY"]: row["GOLDEN_SQL"] for row in rows}


_queries = load_test_queries()
TEST_QUERIES_ANALYST: list[str] = _queries["analyst"]
TEST_QUERIES_SEARCH: list[str] = _queries["search"]
TEST_QUERIES_MIXED: list[str] = _queries["mixed"]

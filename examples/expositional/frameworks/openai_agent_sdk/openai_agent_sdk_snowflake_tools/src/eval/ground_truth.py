"""Ground truth data and test query suites.

Ground truth is loaded from Snowflake tables:
- GROUND_TRUTH_SEARCH: expected retrieval chunks per query
- GROUND_TRUTH_ANALYST: golden SQL per query

Test query lists remain as constants (they are inputs, not ground truth).
"""

from __future__ import annotations

from snowflake.snowpark import Session


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


TEST_QUERIES_ANALYST = [
    "How many tickets are there by priority?",
    "What is the average resolution time for technical tickets?",
    "Which agent has the highest CSAT score?",
    "How many tickets are currently open or pending?",
    "What is the average first response time for urgent tickets?",
    "Which agents resolve tickets the fastest on average, and what is their average CSAT score?",
    "What is the month-over-month trend in ticket volume?",
    "Show me the resolution rate and average CSAT for each ticket category.",
    "Which tickets took longer to resolve than the average for their category?",
    "What percentage of tickets by priority had a first response within one hour?",
    "Who are the top 3 customers by ticket volume and what is their average satisfaction?",
    "Show each agent's total workload, unresolved ticket count, and how many high or urgent tickets they handle.",
    "What is the average days to resolution by month and priority for resolved tickets?",
]

TEST_QUERIES_SEARCH = [
    "How do I reset my password?",
    "What are the API rate limits?",
    "How do I set up SSO?",
    "I can't log in to my account",
    "What is the pricing model?",
]

TEST_QUERIES_MIXED = [
    "How many high priority tickets do we have and what's our SLA?",
    "What is the average CSAT score and how do customers reset passwords?",
    "How do I export my data?",
    "What is the resolution rate for billing tickets?",
    "How do I configure webhooks?",
]

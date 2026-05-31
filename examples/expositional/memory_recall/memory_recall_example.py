"""
Memory Recall Evaluation Example
=================================

This example demonstrates how to use GroundTruthAgreement's memory recall
methods to evaluate an agent memory system's retrieval quality.

Memory recall evaluates whether the right stored memories surface when
needed — distinct from RAG context retrieval metrics.

Setup:
    pip install trulens-feedback trulens-providers-openai
"""

from trulens.feedback import GroundTruthAgreement


# ============================================================
# 1. Define ground truth with expected memories
# ============================================================
#
# Ground truth entries include "expected_memories" (list of memory texts)
# and optionally "conversation_id" to scope lookups.

golden_set = [
    {
        "query": "What issues has the customer reported before?",
        "expected_memories": [
            "Customer reported slow loading on mobile app",
            "Customer complained about missing order #4521",
            "Customer requested refund for damaged item",
        ],
        "conversation_id": "conv_customer_support",
    },
    {
        "query": "Why was the order delayed?",
        "expected_memories": [
            "Warehouse relocation caused shipping delays in March",
            "Order #4521 was flagged for address verification",
        ],
        "conversation_id": "conv_customer_support",
    },
    {
        "query": "What decisions were made about the deadline?",
        "expected_memories": [
            "Team agreed to extend deadline by two weeks",
            "QA requested additional testing time",
        ],
        "conversation_id": "conv_project_planning",
    },
]


# ============================================================
# 2. Create GroundTruthAgreement with conversation_id
# ============================================================

from trulens.providers.openai import OpenAI

gta = GroundTruthAgreement(
    golden_set,
    provider=OpenAI(),
    conversation_id="conv_customer_support",
)


# ============================================================
# 3. Evaluate memory recall
# ============================================================

print("=== Memory Recall Evaluation ===\n")

# Good retrieval: finds most relevant memories
recall_good = gta.memory_recall(
    prompt="What issues has the customer reported before?",
    retrieved_memories=[
        "Customer reported slow loading on mobile app",
        "Customer complained about missing order #4521",
        "Customer asked about loyalty program",  # irrelevant
    ],
)
print(f"Good retrieval recall: {recall_good:.3f}")

# Bad retrieval: returns irrelevant memories
recall_bad = gta.memory_recall(
    prompt="What issues has the customer reported before?",
    retrieved_memories=[
        "Team agreed to extend deadline by two weeks",
        "Customer asked about loyalty program",
    ],
)
print(f"Bad retrieval recall:  {recall_bad:.3f}")

# No ground truth for this query in this conversation
recall_missing = gta.memory_recall(
    prompt="What decisions were made about the deadline?",
    retrieved_memories=["Team agreed to extend deadline by two weeks"],
)
print(f"Missing from conv:     {recall_missing}")  # np.nan


# ============================================================
# 4. Evaluate MRR (ranking quality)
# ============================================================

print("\n=== Memory MRR Evaluation ===\n")

# Relevant memory at rank 1
mrr_top = gta.memory_mrr(
    prompt="What issues has the customer reported before?",
    retrieved_memories=[
        "Customer reported slow loading on mobile app",
        "Customer asked about loyalty program",
    ],
)
print(f"Relevant at rank 1: {mrr_top:.3f}")

# Relevant memory at rank 3
mrr_low = gta.memory_mrr(
    prompt="What issues has the customer reported before?",
    retrieved_memories=[
        "Customer asked about loyalty program",
        "Team agreed to extend deadline by two weeks",
        "Customer reported slow loading on mobile app",
    ],
)
print(f"Relevant at rank 3: {mrr_low:.3f}")


# ============================================================
# 5. Fuzzy matching with similarity_threshold
# ============================================================

print("\n=== Fuzzy Matching ===\n")

# Exact match fails when text is slightly different
recall_exact = gta.memory_recall(
    prompt="Why was the order delayed?",
    retrieved_memories=[
        "Warehouse relocation caused shipping delays",  # truncated
        "Order #4521 was flagged for address verification",
    ],
    similarity_threshold=1.0,
)
print(f"Exact match recall:    {recall_exact:.3f}")

# Fuzzy match catches near-matches
recall_fuzzy = gta.memory_recall(
    prompt="Why was the order delayed?",
    retrieved_memories=[
        "Warehouse relocation caused shipping delays",
        "Order #4521 was flagged for address verification",
    ],
    similarity_threshold=0.7,
)
print(f"Fuzzy match recall:    {recall_fuzzy:.3f}")


# ============================================================
# 6. Using as a TruLens feedback function
# ============================================================

print("\n=== As a TruLens Metric ===\n")

from trulens.core import Metric, Selector

feedback = Metric(
    implementation=gta.memory_recall,
    name="Memory Recall",
    selectors={
        "prompt": Selector.select_record_input(),
        "retrieved_memories": Selector.select_record_output(),
    },
)

print(f"Metric created: {feedback}")
print("Ready to use with TruSession.run_feedback()")

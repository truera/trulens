"""
MemoryRecallMetric Example
===========================

This example demonstrates how to use MemoryRecallMetric to evaluate
an agent memory system's recall quality, including:

1. Basic precision/recall/MRR evaluation
2. Conversation-scoped evaluation with conversation_id
3. Chain completeness for multi-hop reasoning queries
4. Detecting a memory system that retrieves irrelevant context

Setup:
    pip install trulens-feedback
"""

from trulens.feedback.memory_recall import MemoryRecallMetric


# ============================================================
# 1. Define ground truth for a conversation
# ============================================================
#
# Ground truth maps conversation_id -> {memory_id -> {query, relevant IDs, optional chain}}
# This is typically created from a benchmark dataset or manual annotation.

ground_truth = {
    "conv_customer_support": {
        "mem_001": {
            "query": "What issues has the customer reported before?",
            "relevant": ["mem_001", "mem_002", "mem_003"],
            "chain": ["mem_001", "mem_002", "mem_003"],  # Ordered reasoning chain
        },
        "mem_002": {
            "query": "Why was the order delayed?",
            "relevant": ["mem_002", "mem_004"],
            "chain": ["mem_004", "mem_002"],  # Cause -> Effect
        },
    },
    "conv_project_planning": {
        "mem_010": {
            "query": "What decisions were made about the deadline?",
            "relevant": ["mem_010", "mem_011", "mem_012"],
            "chain": ["mem_011", "mem_010", "mem_012"],  # Decision chain
        },
    },
}

metric = MemoryRecallMetric(ground_truth=ground_truth, k=5)


# ============================================================
# 2. Evaluate a single retrieval
# ============================================================

print("=== Single Query Evaluation ===\n")

# Good retrieval: finds relevant memories
result_good = metric.evaluate(
    conversation_id="conv_customer_support",
    query="What issues has the customer reported before?",
    retrieved_ids=["mem_001", "mem_002", "mem_005", "mem_003"],
)
print("Good retrieval result:")
for k, v in result_good.items():
    print(f"  {k}: {v:.3f}")

print()

# Bad retrieval: returns irrelevant memories
result_bad = metric.evaluate(
    conversation_id="conv_customer_support",
    query="What issues has the customer reported before?",
    retrieved_ids=["mem_010", "mem_011", "mem_020", "mem_030", "mem_040"],
)
print("Bad retrieval result:")
for k, v in result_bad.items():
    print(f"  {k}: {v:.3f}")


# ============================================================
# 3. Evaluate an entire conversation
# ============================================================

print("\n=== Conversation-Level Evaluation ===\n")

conversation_queries = [
    ("What issues has the customer reported before?", ["mem_001", "mem_002", "mem_005"]),
    ("Why was the order delayed?", ["mem_002", "mem_004", "mem_001"]),
]

conv_result = metric.evaluate_conversation(
    conversation_id="conv_customer_support",
    queries_and_results=conversation_queries,
)
print("Conversation-level averages:")
for k, v in conv_result.items():
    print(f"  {k}: {v:.3f}")


# ============================================================
# 4. Detecting memory recall issues
# ============================================================

print("\n=== Detecting Memory Issues ===\n")

# Simulate a memory system that only retrieves recent memories,
# missing the earlier ones in a reasoning chain
partial_retrieval = metric.evaluate(
    conversation_id="conv_customer_support",
    query="What issues has the customer reported before?",
    retrieved_ids=["mem_003"],  # Only the last in the chain
)

full_retrieval = metric.evaluate(
    conversation_id="conv_customer_support",
    query="What issues has the customer reported before?",
    retrieved_ids=["mem_001", "mem_002", "mem_003"],  # Full chain
)

print("Partial chain retrieved (only endpoint):")
print(f"  chain_completeness: {partial_retrieval['chain_completeness']:.3f}")
print(f"  recall_at_k:        {partial_retrieval['recall_at_k']:.3f}")

print("\nFull chain retrieved:")
print(f"  chain_completeness: {full_retrieval['chain_completeness']:.3f}")
print(f"  recall_at_k:        {full_retrieval['recall_at_k']:.3f}")

# Key insight: chain_completeness captures that partial retrieval
# breaks the reasoning chain, even if recall_at_k is nonzero.
# This is important for agents that need to answer "why" questions
# by tracing through a sequence of related memories.

print("\n---")
print("chain_completeness is low when only the endpoint of a reasoning")
print("chain is retrieved. This indicates the memory system cannot support")
print("logical reasoning about how events connect.")

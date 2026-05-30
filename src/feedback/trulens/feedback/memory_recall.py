"""Memory recall evaluation metrics for agent memory systems.

Provides MemoryRecallMetric for evaluating how well an agent's memory store
retrieves relevant memories, with support for conversation-scoped evaluation
and multi-hop chain completeness.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pydantic

from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)


class MemoryRecallMetric(pydantic.BaseModel):
    """Evaluates memory recall quality for agent memory systems.

    Takes a ground-truth mapping of expected memory IDs per conversation
    and computes retrieval quality metrics against what the memory store
    actually returned.

    Unlike GroundTruthAgreement which evaluates response quality against
    golden answers, this metric evaluates *retrieval* quality — whether
    the right memories surfaced at the right time.

    Key metrics:
        - precision_at_k: Fraction of retrieved memories that are relevant
        - recall_at_k: Fraction of relevant memories that were retrieved
        - mrr: Reciprocal rank of the first relevant memory
        - chain_completeness: Fraction of expected reasoning chain retrieved

    Usage:
        ```python
        from trulens.feedback.memory_recall import MemoryRecallMetric

        ground_truth = {
            "conv_abc": {
                "mem_1": {"query": "user preferences", "relevant": ["mem_1", "mem_2"]},
                "mem_2": {"query": "past discussions", "relevant": ["mem_3", "mem_4"]},
            }
        }

        metric = MemoryRecallMetric(ground_truth=ground_truth, k=10)

        # Evaluate a single retrieval
        result = metric.evaluate(
            conversation_id="conv_abc",
            query="user preferences",
            retrieved_ids=["mem_1", "mem_5", "mem_3"],
        )
        # => {"precision_at_k": 0.33, "recall_at_k": 0.5, "mrr": 1.0, "chain_completeness": 0.0}
        ```
    """

    ground_truth: Dict[str, Dict[str, Dict]]
    """Ground truth mapping: conversation_id -> {memory_id -> {"query": str, "relevant": [memory_ids], "chain": Optional[[memory_ids]]}}"""

    k: int = 10
    """Number of top results to consider for precision/recall."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def evaluate(
        self,
        conversation_id: str,
        query: str,
        retrieved_ids: List[str],
    ) -> Dict[str, float]:
        """Evaluate memory recall for a single query within a conversation.

        Args:
            conversation_id: The conversation identifier for ground truth lookup.
            query: The query string used to retrieve memories.
            retrieved_ids: List of memory IDs returned by the memory store,
                ordered by relevance (most relevant first).

        Returns:
            Dictionary with keys: precision_at_k, recall_at_k, mrr,
            chain_completeness.
        """
        conv_gt = self.ground_truth.get(conversation_id, {})

        # Find the best matching ground truth entry for this query
        relevant_ids = self._find_relevant_ids(conv_gt, query)
        chain_ids = self._find_chain_ids(conv_gt, query)

        precision = self._precision_at_k(retrieved_ids, relevant_ids, self.k)
        recall = self._recall_at_k(retrieved_ids, relevant_ids, self.k)
        mrr_val = self._mrr(retrieved_ids, relevant_ids)
        chain_comp = self._chain_completeness(retrieved_ids, chain_ids)

        return {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "mrr": mrr_val,
            "chain_completeness": chain_comp,
        }

    def evaluate_conversation(
        self,
        conversation_id: str,
        queries_and_results: List[Tuple[str, List[str]]],
    ) -> Dict[str, float]:
        """Evaluate memory recall across multiple queries in a conversation.

        Args:
            conversation_id: The conversation identifier for ground truth lookup.
            queries_and_results: List of (query, retrieved_ids) tuples.

        Returns:
            Averaged metrics across all queries in the conversation.
        """
        all_metrics = []
        for query, retrieved_ids in queries_and_results:
            m = self.evaluate(conversation_id, query, retrieved_ids)
            all_metrics.append(m)

        if not all_metrics:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "chain_completeness": 0.0,
            }

        return {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0]
        }

    def _find_relevant_ids(
        self, conv_gt: Dict, query: str
    ) -> Set[str]:
        """Find relevant memory IDs for a query from ground truth."""
        relevant = set()
        for mem_id, entry in conv_gt.items():
            if entry.get("query") == query:
                relevant.update(entry.get("relevant", []))
            # Also check if query matches the memory content itself
            if mem_id in entry.get("relevant", []):
                relevant.add(mem_id)
        return relevant

    def _find_chain_ids(
        self, conv_gt: Dict, query: str
    ) -> List[str]:
        """Find reasoning chain memory IDs for a query from ground truth."""
        for mem_id, entry in conv_gt.items():
            if entry.get("query") == query and "chain" in entry:
                return entry["chain"]
        return []

    @staticmethod
    def _precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """Compute Precision@k."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for mid in top_k if mid in relevant_ids)
        return hits / len(top_k)

    @staticmethod
    def _recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """Compute Recall@k."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for mid in top_k if mid in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def _mrr(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        if not relevant_ids:
            return 0.0
        for i, mid in enumerate(retrieved_ids):
            if mid in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _chain_completeness(
        retrieved_ids: List[str],
        chain_ids: List[str],
    ) -> float:
        """Compute chain completeness: fraction of reasoning chain retrieved.

        A reasoning chain is an ordered sequence of memories where each
        link is logically connected (causal, temporal, conditional, etc.).
        Chain completeness measures whether the retrieval returned enough
        of the chain to support logical reasoning.

        Returns:
            Fraction of chain memories found in retrieved results.
            0.0 if no chain is defined. 1.0 if all chain memories are found.
        """
        if not chain_ids:
            return 0.0
        retrieved_set = set(retrieved_ids)
        found = sum(1 for mid in chain_ids if mid in retrieved_set)
        return found / len(chain_ids)

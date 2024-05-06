import logging
import random
import time
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

log = logging.getLogger(__name__)
"""score passages with feedback function, retrying if feedback function fails.
    Args: df: dataframe with columns 'query_id', 'query', 'passage', 'is_selected'
            feedback_func: function that takes query and passage as input and returns a score
            backoff_time: time to wait between retries
            n: number of samples to estimate conditional probabilities of feedback_func's scores
"""


def score_passages(
    df,
    feedback_func_name,
    feedback_func,
    backoff_time=0.5,
    n=5,
    temperature=0.0
):
    grouped = df.groupby('query_id')
    scores = []
    true_relevance = []

    for name, group in grouped:
        query_scores = []
        query_relevance = []
        for _, row in group.iterrows():
            sampled_score = None
            if feedback_func_name == 'TruEra' or n == 1:
                sampled_score = feedback_func(
                    row['query'], row['passage'], temperature
                )  # hard-coded for now, we don't need to sample for TruEra BERT-based model
                time.sleep(backoff_time)
            else:
                sampled_scores = []
                for _ in range(n):
                    sampled_scores.append(
                        feedback_func(
                            row['query'], row['passage'], temperature
                        )
                    )
                    time.sleep(backoff_time)
                sampled_score = sum(sampled_scores) / len(sampled_scores)
            query_scores.append(sampled_score)
            query_relevance.append(row['is_selected'])
            # print(f"Feedback avg score for query {name} is {sampled_score}, is_selected is {row['is_selected']}")

        print(
            f"Feedback function {name} scored {len(query_scores)} out of {len(group)} passages."
        )
        scores.append(query_scores)
        true_relevance.append(query_relevance)

    return scores, true_relevance


def compute_ndcg(scores, true_relevance):
    ndcg_values = [
        ndcg_score([true], [pred])
        for true, pred in zip(true_relevance, scores)
    ]
    return np.mean(ndcg_values)


def compute_ece(scores, true_relevance, n_bins=10):
    ece = 0
    for bin in np.linspace(0, 1, n_bins):
        bin_scores = []
        bin_truth = []
        for score_list, truth_list in zip(scores, true_relevance):
            for score, truth in zip(score_list, truth_list):
                if bin <= score < bin + 1 / n_bins:
                    bin_scores.append(score)
                    bin_truth.append(truth)

        if bin_scores:
            bin_avg_confidence = np.mean(bin_scores)
            bin_accuracy = np.mean(bin_truth)
            ece += np.abs(bin_avg_confidence - bin_accuracy
                         ) * len(bin_scores) / sum(map(len, scores))

    return ece


def precision_at_k(scores, true_relevance, k):
    sorted_scores = sorted(scores, reverse=True)
    kth_score = sorted_scores[min(k - 1, len(scores) - 1)]

    # Indices of items with scores >= kth highest score
    top_k_indices = [i for i, score in enumerate(scores) if score >= kth_score]

    # Calculate precision
    true_positives = sum(np.take(true_relevance, top_k_indices))
    return true_positives / len(top_k_indices) if top_k_indices else 0


def recall_at_k(scores, true_relevance, k):
    """
    Calculate the recall at K.

    Parameters:
    true_relevance (list of int): List of binary values indicating relevance (1 for relevant, 0 for not).
    scores (list of float): List of scores assigned by the model.
    k (int): Number of top items to consider for calculating recall.

    Returns:
    float: Recall at K.
    """
    sorted_scores = sorted(scores, reverse=True)
    kth_score = sorted_scores[min(k - 1, len(scores) - 1)]

    # Indices of items with scores >= kth highest score
    top_k_indices = [i for i, score in enumerate(scores) if score >= kth_score]

    # Calculate recall
    relevant_indices = np.where(true_relevance)[0]
    hits = sum(idx in top_k_indices for idx in relevant_indices)
    total_relevant = sum(true_relevance)

    return hits / total_relevant if total_relevant > 0 else 0

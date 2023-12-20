import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from typing import List

def score_passages(df, feedback_func):
    grouped = df.groupby('query_id')
    scores = []
    true_relevance = []
    
    for name, group in grouped:
        query_scores = []
        query_relevance = []
        for _, row in group.iterrows():
            score = feedback_func(row['query'], row['passage'])
            query_scores.append(score)
            query_relevance.append(row['is_selected'])
        
        scores.append(query_scores)
        true_relevance.append(query_relevance)

    return scores, true_relevance

def compute_ndcg(scores, true_relevance):
    ndcg_values = [ndcg_score([true], [pred]) for true, pred in zip(true_relevance, scores)]
    return np.mean(ndcg_values)

def compute_ece(scores, true_relevance, n_bins=10):
    ece = 0
    for bin in np.linspace(0, 1, n_bins):
        bin_scores = []
        bin_truth = []
        for score_list, truth_list in zip(scores, true_relevance):
            for score, truth in zip(score_list, truth_list):
                if bin <= score < bin + 1/n_bins:
                    bin_scores.append(score)
                    bin_truth.append(truth)
        
        if bin_scores:
            bin_avg_confidence = np.mean(bin_scores)
            bin_accuracy = np.mean(bin_truth)
            ece += np.abs(bin_avg_confidence - bin_accuracy) * len(bin_scores) / sum(map(len, scores))

    return ece

def precision_at_k(true_relevance, scores, k):
    top_k_indices = np.argsort(scores)[-k:]
    
    # Calculate precision
    true_positives = sum(np.take(true_relevance, top_k_indices))
    return true_positives / k

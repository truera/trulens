import numpy as np

# Example data
scores = [[0.9, 0.7, 0.4], [0.8, 0.3, 0.1]]
true_relevance = [[1, 0, 0], [0, 0, 1]]  # Binary relevance (1 for relevant, 0 for not relevant)
n_bins = 3

def compute_ece(scores, true_relevance, n_bins=3):
    ece = 0
    for bin in np.linspace(0, 1, n_bins):
        bin_scores = []
        bin_truth = []
        for score_list, truth_list in zip(scores, true_relevance):
            print('score list: ', score_list)
            print('truth list: ', truth_list)
            for score, truth in zip(score_list, truth_list):
                if bin <= score < bin + 1/n_bins:
                    bin_scores.append(score)
                    bin_truth.append(truth)

        if bin_scores:
            print('bin scores: ', bin_scores)
            print('bin truth: ', bin_truth)
            bin_avg_confidence = np.mean(bin_scores)
            print('bin avg confidence: ', bin_avg_confidence)
            bin_accuracy = np.mean(bin_truth)
            print('bin accuracy: ', bin_accuracy)
            ece += np.abs(bin_avg_confidence - bin_accuracy) * len(bin_scores) / sum(map(len, scores))

    return ece

# Compute ECE for the example data
compute_ece(scores, true_relevance, n_bins)


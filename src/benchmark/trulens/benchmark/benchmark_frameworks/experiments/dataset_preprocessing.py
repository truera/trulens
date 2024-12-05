import ast
from collections import defaultdict
import csv
import json
import random
from typing import Any, List, Tuple

from datasets import load_dataset
import ir_datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from trulens.feedback import GroundTruthAggregator


def generate_balanced_llm_aggrefact_benchmark(split="test", random_seed=42):
    llm_aggrefact_dataset = load_dataset("lytang/LLM-AggreFact")

    # Convert to pandas DataFrame
    df = pd.DataFrame(llm_aggrefact_dataset[split])

    # Initialize an empty list to store balanced DataFrames
    balanced_dfs = []

    # Iterate over each unique dataset
    for dataset_name in df["dataset"].unique():
        # Filter the DataFrame for the current dataset
        df_subset = df[df["dataset"] == dataset_name]

        # Count the number of instances for each class
        class_counts = df_subset["label"].value_counts()

        # Determine the minimum count between the two classes
        min_count = class_counts.min()

        # Sample min_count instances from each class
        df_balanced = (
            df_subset.groupby("label")
            .apply(lambda x: x.sample(min_count, random_state=random_seed))
            .reset_index(drop=True)
        )

        # Append the balanced DataFrame to the list
        balanced_dfs.append(df_balanced)

    # Concatenate all balanced DataFrames into a final DataFrame
    final_balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Display the balanced DataFrame
    return final_balanced_df


def generate_summeval_groundedness_golden_set(
    file_path: str, max_samples_per_bucket: int = 200
):
    """
    Generate a balanced groundedness golden set from the Summeval dataset.

    Args:
        file_path (str): Path to the JSON file.
        max_samples_per_bucket (int): Maximum number of samples per score bucket.

    Yields:
        dict: A dictionary containing query, expected_response, expected_score, and human_score.
    """

    def calculate_expected_score(
        normalized_metrics_lst: List[Any], weights_lst: List[float]
    ):
        """Calculate the expected score using normalized metrics and weights."""
        assert len(normalized_metrics_lst) == len(weights_lst)
        return round(
            sum(
                normalized_metrics_lst[i] * weights_lst[i]
                for i in range(len(normalized_metrics_lst))
            )
            / sum(weights_lst),
            2,
        )

    # Track the number of samples in each bucket
    buckets = {
        "low": 0,  # Scores between 0.0 and 0.3
        "medium": 0,  # Scores between 0.3 and 0.7
        "high": 0,  # Scores between 0.7 and 1.0
    }

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Parse each line as a JSON object
            try:
                data = json.loads(line)

                # Ensure the expected keys exist in the JSON
                try:
                    row = data
                    assert (
                        len(row["machine_summaries"]) == len(row["consistency"])
                    ), "Mismatch in lengths of machine_summaries and consistency"

                    # Iterate over the summaries and create the desired dictionary structure
                    for i in range(len(row["machine_summaries"])):
                        query = row.get("text", "")
                        expected_response = row["machine_summaries"][i]

                        # Calculate the expected score based on normalized consistency score
                        expected_score = calculate_expected_score(
                            [
                                (row["consistency"][i] - 1)
                                / 4,  # Normalize from [1, 5] to [0, 1]
                            ],
                            [1.0],
                        )

                        # Determine the bucket based on the score range
                        if expected_score <= 0.3:
                            bucket = "low"
                        elif expected_score <= 0.7:
                            bucket = "medium"
                        else:
                            bucket = "high"

                        # Yield the record only if the bucket has not reached the limit
                        if buckets[bucket] < max_samples_per_bucket:
                            buckets[bucket] += 1
                            yield {
                                "query": query,
                                "expected_response": expected_response,
                                "expected_score": expected_score,
                                "human_score": row["consistency"][i],
                            }

                except KeyError as e:
                    print(
                        f"Key error: {e}. Please check if the keys exist in the JSON file."
                    )
                except AssertionError as e:
                    print(
                        f"Assertion error: {e}. The lengths of 'machine_summaries' and 'consistency' do not match."
                    )

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}. Check the line format.")


# Snowflake IT dataset


def generate_snowflake_it_golden_set_groundedness(file_path: str):
    res = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        for row in all_rows:
            # Convert the 'golden' from a string to a list
            try:
                expected_chunks = ast.literal_eval(row["golden"])
                if not isinstance(expected_chunks, list):
                    raise ValueError("Golden column should be a list")

                for expected_chunk in expected_chunks:
                    # Yield the required fields
                    res.append({
                        "query": expected_chunk,  # source
                        "expected_response": row[
                            "expected_response"
                        ],  # statement
                        "expected_score": 1,  # retrieved chunks in the "golden" colum are always considered grounded
                    })

                # Generate a negative example for each query
                # Collect all possible chunks from other queries to use as negative contexts
                other_chunks = [
                    chunk
                    for other_row in all_rows
                    if other_row != row
                    for chunk in ast.literal_eval(other_row["golden"])
                ]

                # Randomly select a negative chunk (context from another query)
                if other_chunks:
                    negative_chunk = random.choice(other_chunks)
                    res.append({
                        "query": negative_chunk,
                        "expected_response": row[
                            "expected_response"
                        ],  # statement (not grounded by the chunk)
                        "expected_score": 0,  # Negative example, score = 0
                    })

            except (ValueError, SyntaxError) as e:
                print(f"Error parsing golden column: {e}")
                continue

    return res


def generate_snowflake_it_golden_set_answer_relevance(file_path: str):
    res = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(
            reader
        )  # Store all rows in memory to use for negative example generation

        for row in all_rows:
            # generate a positive example for each query
            if (
                "I don’t know the answer to that question."
                in row["expected_response"]
            ):
                ground_truth_score = 0  # label answer relevance as 0 for ABSTENTION "I don’t know the answer to that question."
            else:
                ground_truth_score = (
                    1  # label answer relevance as 1 for all other cases
                )
            res.append({
                "query": row["query"],
                "expected_response": row["expected_response"],
                "expected_score": ground_truth_score,
            })

            # generate an easy negative example for each positive example by randomly selecting a response from another query
            negative_response = random.choice([
                r["expected_response"] for r in all_rows if r != row
            ])
            res.append({
                "query": row["query"],
                "expected_response": negative_response,  # Orthogonal response
                "expected_score": 0,  # Label answer relevance as 0 for negative examples
            })

    return res


def generate_snowflake_it_golden_set_context_relevance(file_path: str):
    res = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

        # Step 1: Process each row to extract positive examples
        for row in all_rows:
            try:
                expected_chunks = ast.literal_eval(row["golden"])

                if not isinstance(expected_chunks, list):
                    raise ValueError("Golden column should be a list")

                # Generate positive examples
                for expected_chunk in expected_chunks:
                    res.append({
                        "query": row["query"],
                        "expected_response": expected_chunk,
                        "expected_score": 1,  # Positive example, score = 1
                    })

                # Step 2: Generate a negative example for each query
                # Collect all possible chunks from other queries to use as negative contexts
                other_chunks = [
                    chunk
                    for other_row in all_rows
                    if other_row != row
                    for chunk in ast.literal_eval(other_row["golden"])
                ]

                # Randomly select a negative chunk (context from another query)
                if other_chunks:
                    negative_chunk = random.choice(other_chunks)
                    res.append({
                        "query": row["query"],
                        "expected_response": negative_chunk,  # Orthogonal/negative context
                        "expected_score": 0,  # Negative example, score = 0
                    })

            except (ValueError, SyntaxError) as e:
                print(
                    f"Error parsing golden column for query '{row['query']}': {e}"
                )
                continue

        return res


def generate_qags_golden_set_groundedness(
    file_path: str, max_samples_per_bucket: int = 100
):
    # Initialize counters for score ranges
    buckets = {
        "low": 0,  # Scores between 0.0 and 0.3
        "medium": 0,  # Scores between 0.3 and 0.7
        "high": 0,  # Scores between 0.7 and 1.0
    }

    # Open the file and iterate through each line
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Parse each line as a JSON object
            data = json.loads(line)

            # Extract the article as the query
            query = data["article"]

            # Iterate over the summary_sentences to flatten the structure
            for summary in data["summary_sentences"]:
                expected_response = summary["sentence"]

                # Calculate expected_score based on worker responses
                responses = [
                    response["response"] for response in summary["responses"]
                ]
                # Convert 'yes' to 1 and 'no' to 0, then calculate the average score
                expected_score = sum(
                    1 if r.lower() == "yes" else 0 for r in responses
                ) / len(responses)

                # Determine the bucket based on the score range
                if expected_score <= 0.3:
                    bucket = "low"
                elif expected_score <= 0.7:
                    bucket = "medium"
                else:
                    bucket = "high"

                # Yield the record only if the bucket has not reached the limit
                if buckets[bucket] < max_samples_per_bucket:
                    buckets[bucket] += 1
                    yield {
                        "query": query,
                        "expected_response": expected_response,
                        "expected_score": expected_score,
                    }


def generate_ms_marco_context_relevance_benchmark(
    file_path: str = "data/ms_marco_v2_1_val.parquet",
):
    df = pd.read_parquet(file_path, engine="pyarrow")  # or engine='fastparquet'

    for _, row in df.iterrows():
        assert len(row["passages"]["is_selected"]) == len(
            row["passages"]["passage_text"]
        )

        if sum(row["passages"]["is_selected"]) < 1:
            # currently we only consider sample with one passage marked as relevant (there are samples where zero passage_text is selected)
            continue
        for i, passage_text in enumerate(row["passages"]["passage_text"]):
            yield {
                "query_id": row["query_id"],
                "query": row["query"],
                "expected_response": passage_text,
                "expected_score": row["passages"]["is_selected"][
                    i
                ],  # Binary relevance
            }


def generate_balanced_ms_marco_hard_negatives_dataset(
    series: pd.Series, sample_size: int = 400
):
    # sampled_series = series.sample(n=sample_size, random_state=42)
    sampled_series = series[:sample_size]
    # Step 2: Create a list for the balanced dataset
    balanced_dataset = []

    # Step 3: Iterate over the sampled rows
    for idx, row in sampled_series.items():
        # "row" is a dictionary containing 'query', 'positive', and 'negative'
        query = row.get("query")
        positive_list = row.get("positive", [])
        negative_list = row.get("negative", [])
        print(f"query: {query}")
        # Select one positive example
        if positive_list:
            positive_example = random.choice(positive_list)
        else:
            continue  # Skip if no positive examples

        # Select one negative example
        if negative_list:
            negative_example = random.choice(negative_list)
        else:
            continue  # Skip if no negative examples

        print(
            f"positive_example: {positive_example} \n negative_example: {negative_example}"
        )
        # Add a positive example to the dataset
        balanced_dataset.append({
            "query": query,
            "expected_response": positive_example,
            "expected_score": 1,  # Positive example, label 1
        })

        # Add a negative example to the dataset
        balanced_dataset.append({
            "query": query,
            "expected_response": negative_example,
            "expected_score": 0,  # Negative example, label 0
        })


def generate_trec_dl_passage_benchmark(
    max_samples_per_query_per_score: int = 3,
    dataset_path: str = "msmarco-passage-v2/trec-dl-2021/judged",
):
    # Combine queries and qrels from multiple datasets
    queries = {}
    qrels = defaultdict(dict)
    docs_store = None

    dataset = ir_datasets.load(dataset_path)
    # Merge queries
    queries.update({q.query_id: q for q in dataset.queries_iter()})
    # Merge qrels
    for qid, docs in dataset.qrels_dict().items():
        qrels[qid].update(docs)
    # Get docs_store
    if docs_store is None:
        docs_store = dataset.docs_store()

    print("Total number of queries:", len(queries))
    print("Total number of qrels:", len(qrels))

    # Sampling
    for query_id, query in queries.items():
        if query_id not in qrels:
            print("query_id not found in qrels")
            continue  # Skip queries without relevance judgments

        # Get documents by relevance scores
        relevant_docs = defaultdict(list)
        for doc_id, score in qrels[query_id].items():
            relevant_docs[score].append(doc_id)

        # Determine scoreddocs intervals for this query
        scored_docs = [
            scored_doc
            for scored_doc in ir_datasets.load(dataset_path).scoreddocs_iter()
            if scored_doc.query_id == query_id
        ]
        if not scored_docs:
            continue

        min_score = min(scored_doc.score for scored_doc in scored_docs)
        max_score = max(scored_doc.score for scored_doc in scored_docs)
        interval_size = (max_score - min_score) / 4
        intervals = [
            (min_score + i * interval_size, min_score + (i + 1) * interval_size)
            for i in range(4)
        ]

        # Initialize sampling counts
        sampled_docs = []

        # Use scoreddocs for all scores (0, 1, 2, and 3)
        for score in [0, 1, 2, 3]:
            if score in relevant_docs:
                # Get ranked documents using scoreddocs
                ranked_docs = []
                for scored_doc in scored_docs:
                    if scored_doc.doc_id in relevant_docs[score]:
                        ranked_docs.append((
                            scored_doc.doc_id,
                            scored_doc.score,
                        ))

                # Filter documents based on interval alignment (-1, 0, +1)
                allowed_intervals = [
                    intervals[max(0, score - 1)],
                    intervals[score],
                    intervals[min(3, score + 1)],
                ]
                interval_docs = [
                    (doc_id, doc_score)
                    for doc_id, doc_score in ranked_docs
                    if any(
                        low <= doc_score <= high
                        for low, high in allowed_intervals
                    )
                ]

                # Sort by score (descending) and select top documents
                interval_docs.sort(key=lambda x: x[1], reverse=True)
                top_docs = [
                    doc_id
                    for doc_id, _ in interval_docs[
                        :max_samples_per_query_per_score
                    ]
                ]

                # Add to sampled documents
                sampled_docs.extend(top_docs)

        doc_text_seen = set()  # deduplication of identical passages
        # Yield the sampled data
        for doc_id in sampled_docs:
            doc = docs_store.get(doc_id)
            if doc and doc.text not in doc_text_seen:
                doc_text_seen.add(doc.text)
                yield {
                    "query_id": query_id,
                    "query": query.text,
                    "doc_id": doc_id,
                    "expected_response": doc.text
                    if hasattr(doc, "text")
                    else doc.body,
                    "expected_score": qrels[query_id][doc_id]
                    / 3,  # Normalize to [0, 1]
                }


def write_results(
    feedback_scores: List[float],
    labels: List[Any],
    latencies: List[float],
    file_name: str,
):
    assert len(feedback_scores) == len(labels)

    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(feedback_scores)
        writer.writerow(labels)
        writer.writerow(latencies)


def read_results(
    file_name: str,
) -> Tuple[List[Any], List[Any], List[float]]:
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index == 0:
                # First row contains scores
                scores = list(map(float, row))  # Convert strings to floats
            elif index == 1:
                # Second row contains labels
                labels = list(map(float, row))  # Convert strings to floats
            elif index == 2:
                # Third row contains latencies
                latencies = list(map(float, row))
    return scores, labels, latencies


def visualize_expected_score_distribution(scores: List[float]):
    """
    Visualize the distribution of expected scores in the generated sample set.

    Args:
        scores (List[float]): List of expected scores.
    Returns:
        None: Displays the histogram of the expected score distribution.
    """

    # Plot the histogram of the expected scores
    plt.figure(figsize=(8, 6))
    sns.histplot(scores, bins=10, kde=True, color="skyblue")
    plt.title("Distribution of Expected Scores")
    plt.xlabel("Expected Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(
    true_labels: List[int],
    predicted_labels: List[int],
    threshold=0.5,
    title="Confusion Matrix",
):
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def compute_binary_classification_metrics(
    dataset_name: str,
    binary_labels: List[int],
    generated_scores: List[int],
    latencies: List[float],
) -> None:
    f_recall = GroundTruthAggregator(binary_labels).recall
    f_precision = GroundTruthAggregator(binary_labels).precision
    f_f1_score = GroundTruthAggregator(binary_labels).f1_score
    f_cohens_kappa = GroundTruthAggregator(binary_labels).cohens_kappa
    f_matthews = GroundTruthAggregator(binary_labels).matthews_correlation
    recall = f_recall(generated_scores)
    precision = f_precision(generated_scores)
    f1_score = f_f1_score(generated_scores)
    cohens_kappa = f_cohens_kappa(generated_scores)
    matthews = f_matthews(generated_scores)
    avg_latency = sum(latencies) / len(latencies) if len(latencies) > 0 else 0

    print(
        f"recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1_score:.4f}, Cohen's Kappa: {cohens_kappa:.4f}, Matthews: {matthews:.4f}, avg_latency: {avg_latency:.4f}"
    )
    plot_confusion_matrix(
        binary_labels,
        generated_scores,
        title=f"Confusion Matrix {dataset_name}",
    )

import ast
import csv
import json
import random
from typing import List, Tuple

import pandas as pd


def generate_summeval_groundedness_golden_set(file_path):
    def calculate_expected_score(normalized_metrics_lst, weights_lst):
        assert len(normalized_metrics_lst) == len(weights_lst)
        return round(
            sum(
                normalized_metrics_lst[i] * weights_lst[i]
                for i in range(len(normalized_metrics_lst))
            )
            / sum(weights_lst),
            2,
        )

    with open(file_path) as f:
        for line in f:
            # Each line is a separate JSON object
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
                        yield {
                            "query": row.get(
                                "text", ""
                            ),  # Default to empty string if key not found
                            "expected_response": row["machine_summaries"][i],
                            "expected_score": calculate_expected_score(
                                [
                                    (row["consistency"][i] - 1)
                                    / 4,  # Normalize from [1, 5] to [0, 1]
                                ],
                                [1.0],
                            ),
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


def generatate_snowflake_it_golden_set_groundedness(file_path):
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


def generate_snowflake_it_golden_set_answer_relevance(file_path):
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


def generate_snowflake_it_golden_set_context_relevance(file_path):
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


def generate_qags_golden_set_groundedness(file_path):
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
                # Convert 'yes' to 1 and 'no' to 0, then calculate the average
                expected_score = sum(
                    1 if r.lower() == "yes" else 0 for r in responses
                ) / len(responses)

                # Yield the processed record
                yield {
                    "query": query,
                    "expected_response": expected_response,
                    "expected_score": expected_score,
                }


def generate_ms_marco_context_relevance_benchmark(
    file_path="data/ms_marco_v2_1_val.parquet",
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


def write_results(
    feedback_scores: List[float],
    labels: List[float | int],
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
) -> Tuple[List[float | int], List[float | int], List[float]]:
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

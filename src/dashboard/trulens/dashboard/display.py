import re
import time
from typing import Dict, List

import pandas as pd
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.dashboard.ux.styles import CATEGORY


def get_icon(fdef: feedback_schema.FeedbackDefinition, result: float) -> str:
    """
    Get the icon for a given feedback definition and result.

    Args:
        fdefThe feedback definition.

        result: The result of the feedback.

    Returns:
        str: The icon for the feedback
    """
    cat = CATEGORY.of_score(
        result or 0,
        higher_is_better=fdef.higher_is_better
        if fdef.higher_is_better is not None
        else True,
        is_distance="distance" in fdef.name.lower(),
    )
    return cat.icon


def get_feedback_result(
    tru_record: record_schema.Record, feedback_name: str, timeout: int = 60
) -> pd.DataFrame:
    """
    Retrieve the feedback results including metadata (such as reasons) for a given feedback name from a TruLens record.

    Args:
        tru_record: The record containing feedback and future results.
        feedback_name: The name of the feedback to retrieve results for.

    Returns:
        pd.DataFrame: A DataFrame containing the feedback results. If no feedback
            results are found, an empty DataFrame is returned.
    """
    start_time = time.time()
    feedback_calls = None

    while time.time() - start_time < timeout:
        feedback_calls = next(
            (
                future_result.result()
                for feedback_definition, future_result in tru_record.feedback_and_future_results
                if feedback_definition.name == feedback_name
            ),
            None,
        )
        if feedback_calls is not None:
            break
        time.sleep(1)  # Wait for 1 second before checking again

    if feedback_calls is None:
        raise TimeoutError(
            f"Feedback for '{feedback_name}' not available within {timeout} seconds."
        )

    # Ensure feedback_calls is iterable
    if not hasattr(feedback_calls, "__iter__"):
        raise ValueError("feedback_calls is not iterable")

    feedback_result = [
        {
            **call.model_dump()["args"],
            "score": call.model_dump()["ret"],
            **call.model_dump()["meta"],
        }
        for call in feedback_calls.calls
    ]

    return pd.DataFrame(feedback_result)


def highlight(
    row: pd.Series,
    selected_feedback: str,
    feedback_directions: Dict[str, bool],
    default_direction: str,
) -> List[str]:
    """
    Apply background color to the rows of a DataFrame based on the selected feedback.

    Args:
        row (pandas.Series): A row of the DataFrame to be highlighted.
        selected_feedback (str): The selected feedback to determine the background color.
        feedback_directions (dict): A dictionary mapping feedback names to their directions.
        default_direction (str): The default direction for feedback.

    Returns:
        list: A list of CSS styles representing the background color for each cell in the row.
    """
    if "distance" in selected_feedback:
        return [f"background-color: {CATEGORY.DISTANCE.color}"] * len(row)

    cat = CATEGORY.of_score(
        row["score"],
        higher_is_better=feedback_directions.get(
            selected_feedback, default_direction == "HIGHER_IS_BETTER"
        ),
    )
    # Apply the background color to the entire row
    return [f"background-color: {cat.color}"] * len(row)


def expand_groundedness_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the groundedness DataFrame by splitting the reasons column into separate rows and columns.

    Args:
        df (pd.DataFrame): The groundedness DataFrame.

    Returns:
        pd.DataFrame: The expanded DataFrame.
    """
    # Split the reasons value into separate rows and columns
    if "explanation" in df.columns:
        reasons = df["explanation"].iloc[0]
    elif "reasons" in df.columns:
        reasons = df["reasons"].iloc[0]
    else:
        raise ValueError(
            "Missing 'explanation' or 'reasons' column from feedbacks."
        )
    # Split the reasons into separate statements
    statements = reasons.split("STATEMENT")
    data = []
    # Each reason has three components: statement, supporting evidence, and score
    # Parse each reason into these components and add them to the data list
    for statement in statements[1:]:
        try:
            criteria = statement.split("Criteria: ")[1].split(
                "Supporting Evidence: "
            )[0]
            supporting_evidence = statement.split("Supporting Evidence: ")[
                1
            ].split("Score: ")[0]
            score_pattern = re.compile(r"([0-9]*\.?[0-9]+)(?=\D*$)")
            score_split = statement.split("Score: ")[1]
            score_match = score_pattern.search(score_split)
            if score_match:
                score = float(score_match.group(1))
            else:
                score = None
        except IndexError:
            # Handle cases where the expected substrings are not found
            criteria = None
            supporting_evidence = None
            score = None
        data.append({
            "Statement": criteria,
            "Supporting Evidence from Source": supporting_evidence,
            "Groundedness Score": score,
        })
    reasons_df = pd.DataFrame(data)

    reasons_df.rename(columns={"Groundedness Score": "score"}, inplace=True)

    # Return only the expanded reasons DataFrame
    return reasons_df.reset_index(drop=True)

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

    # Handle None scores
    score = row.get("score")
    if score is None:
        return [f"background-color: {CATEGORY.UNKNOWN.color}"] * len(row)

    cat = CATEGORY.of_score(
        score,
        higher_is_better=feedback_directions.get(
            selected_feedback, default_direction == "HIGHER_IS_BETTER"
        ),
    )
    # Apply the background color to the entire row
    return [f"background-color: {cat.color}"] * len(row)


def expand_groundedness_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the groundedness reasons into a DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing 'reasons' column.

    Returns:
        pd.DataFrame: A DataFrame with expanded groundedness reasons.
    """
    # First check for new list[dict] reasons format (preferred going forward)
    if "reasons" in df.columns and df["reasons"].notna().any():
        first_val = df.loc[df["reasons"].notna(), "reasons"].iloc[0]
        if isinstance(first_val, list):
            reasons_list = first_val
            if not reasons_list:
                return pd.DataFrame({
                    "Statement": [],
                    "Supporting Evidence from Source": [],
                    "score": [],
                })

            reasons_df = pd.DataFrame(reasons_list)
            # Normalize key names
            reasons_df.rename(
                columns={
                    "criteria": "Statement",
                    "supporting_evidence": "Supporting Evidence from Source",
                },
                inplace=True,
            )

            # Ensure expected columns exist
            for col in [
                "Statement",
                "Supporting Evidence from Source",
                "score",
            ]:
                if col not in reasons_df.columns:
                    reasons_df[col] = None

            return reasons_df[
                [
                    "Statement",
                    "Supporting Evidence from Source",
                    "score",
                ]
            ].reset_index(drop=True)

    # -------------------- explanation column as list[str] -------------------------
    if "explanation" in df.columns and df["explanation"].notna().any():
        exp_val = df.loc[df["explanation"].notna(), "explanation"].iloc[0]

        # Try to parse JSON string to list[dict]
        if isinstance(exp_val, str) and exp_val.strip().startswith("["):
            import json

            try:
                parsed = json.loads(exp_val)
                if isinstance(parsed, list):
                    exp_val = parsed
            except Exception:
                pass

        # Case: list of dicts with keys
        if isinstance(exp_val, list) and all(
            isinstance(i, dict) for i in exp_val
        ):
            reasons_df = pd.DataFrame(exp_val)
            reasons_df.rename(
                columns={
                    "criteria": "Statement",
                    "supporting_evidence": "Supporting Evidence from Source",
                },
                inplace=True,
            )
            if "score" not in reasons_df.columns:
                reasons_df["score"] = None
            return reasons_df[
                [
                    "Statement",
                    "Supporting Evidence from Source",
                    "score",
                ]
            ].reset_index(drop=True)

        # Case: list[str] legacy
        if isinstance(exp_val, list):
            rows = []
            for item in exp_val:
                if not isinstance(item, str):
                    item = str(item)
                crit_match = re.search(
                    r"Criteria:\s*(.+?)(?:\n|Supporting Evidence:|$)",
                    item,
                    re.DOTALL,
                )
                criteria = (
                    crit_match.group(1).strip() if crit_match else item.strip()
                )

                sup_match = re.search(
                    r"Supporting Evidence:\s*(.+)", item, re.DOTALL
                )
                evidence = sup_match.group(1).strip() if sup_match else ""

                score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", item)
                score_val = float(score_match.group(1)) if score_match else None

                rows.append({
                    "Statement": criteria,
                    "Supporting Evidence from Source": evidence,
                    "score": score_val,
                })

            return pd.DataFrame(rows).reset_index(drop=True)

    # -------------------- legacy string-based parsing -----------------------------
    reasons = None
    if "explanation" in df.columns and df["explanation"].notna().any():
        reasons = df.loc[df["explanation"].notna(), "explanation"].iloc[0]
    elif "reasons" in df.columns and df["reasons"].notna().any():
        candidate = df.loc[df["reasons"].notna(), "reasons"].iloc[0]
        if isinstance(candidate, str):
            reasons = candidate

    if reasons is None:
        # nothing to expand
        return pd.DataFrame({
            "Statement": [],
            "Supporting Evidence from Source": [],
            "score": [],
        })

    # Convert to string
    if not isinstance(reasons, str):
        reasons = str(reasons)

    statements = re.split(r"\bSTATEMENT \d+:", reasons)[1:]
    data = []
    for s in statements:
        s = s.strip()
        if not s:
            continue
        crit_match = re.search(
            r"Criteria:\s*(.+?)(?:\n|Supporting Evidence:)", s, re.DOTALL
        )
        criteria = crit_match.group(1).strip() if crit_match else ""
        sup_match = re.search(
            r"Supporting Evidence:\s*(.+?)(?:\n|Score:)", s, re.DOTALL
        )
        evidence = sup_match.group(1).strip() if sup_match else ""
        score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", s)
        score_val = float(score_match.group(1)) if score_match else None
        data.append({
            "Statement": criteria,
            "Supporting Evidence from Source": evidence,
            "score": score_val,
        })

    return pd.DataFrame(data).reset_index(drop=True)

from functools import partial
import pprint as pp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
from trulens.core.database.base import MULTI_CALL_NAME_DELIMITER
from trulens.dashboard.display import expand_groundedness_df
from trulens.dashboard.display import highlight
from trulens.dashboard.ux.styles import CATEGORY
from trulens.dashboard.ux.styles import default_direction


def df_cell_highlight(
    score: float,
    feedback_name: str,
    feedback_directions: Dict[str, bool],
    n_cells: int = 1,
) -> List[str]:
    """Returns the background color for a cell in a DataFrame based on the score and feedback name.

    Args:
        score (float): The score value to determine the background color.
        feedback_name (str): The feedback name to determine the background color.
        feedback_directions (dict): A dictionary mapping feedback names to their directions. True if higher is better, False otherwise.
        n_cells (int, optional): The number of cells to apply the background color. Defaults to 1.

    Returns:
        A list of CSS styles representing the background color.
    """
    if "distance" in feedback_name:
        return [f"background-color: {CATEGORY.UNKNOWN.color}"] * n_cells
    cat = CATEGORY.of_score(
        score,
        higher_is_better=feedback_directions.get(
            feedback_name, default_direction == "HIGHER_IS_BETTER"
        ),
    )
    return [f"background-color: {cat.color}"] * n_cells


def _identify_span_types(
    call: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Identify and separate EVAL_ROOT and EVAL spans from the call list.

    Args:
        call: List of call dictionaries containing span information

    Returns:
        Tuple of (eval_root_calls, eval_calls) lists
    """
    eval_root_calls = []
    eval_calls = []

    for c in call:
        # For OTel spans, use explicit span_type field
        if c.get("span_type") == "EVAL_ROOT":
            eval_root_calls.append(c)
        elif c.get("span_type") == "EVAL":
            eval_calls.append(c)
        # For legacy spans (pre-OTel), all calls should contain the following fields: args, ret, and meta
        elif "args" in c and "ret" in c and "meta" in c:
            eval_calls.append(c)

    return eval_root_calls, eval_calls


def _filter_eval_calls_by_root(
    eval_root_calls: List[Dict[str, Any]], eval_calls: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Filter EVAL calls to only those belonging to the most recent EVAL_ROOT spans.

    Args:
        eval_root_calls: List of EVAL_ROOT span dictionaries
        eval_calls: List of EVAL span dictionaries

    Returns:
        Filtered list of EVAL span dictionaries

    Raises:
        KeyError: If eval_root_id is missing from any EVAL_ROOT or EVAL call
    """
    if not eval_root_calls:
        return eval_calls

    eval_root_df = pd.DataFrame(eval_root_calls)
    if "eval_root_id" not in eval_root_df.columns:
        raise KeyError("eval_root_id column missing from EVAL_ROOT spans")

    if eval_root_df.empty:
        return eval_calls

    eval_root_df = _filter_duplicate_span_calls(eval_root_df)
    most_recent_eval_root_ids = set(eval_root_df["eval_root_id"].unique())

    # Verify all eval_calls have eval_root_id
    for c in eval_calls:
        if "eval_root_id" not in c:
            raise KeyError("eval_root_id missing from EVAL spans")

    return [
        c for c in eval_calls if c["eval_root_id"] in most_recent_eval_root_ids
    ]


def _process_eval_calls_for_display(
    eval_calls: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Process EVAL calls into a displayable DataFrame.

    Args:
        eval_calls: List of EVAL span dictionaries

    Returns:
        DataFrame ready for display
    """
    # Convert non-string args to formatted strings
    for c in eval_calls:
        args: Dict = c["args"]
        for k, v in args.items():
            if not isinstance(v, str):
                args[k] = pp.pformat(v)

    # Create DataFrame from processed calls
    df = pd.DataFrame.from_records(c["args"] for c in eval_calls)
    df["score"] = pd.DataFrame([
        float(eval_calls[i]["ret"]) if eval_calls[i]["ret"] is not None else -1
        for i in range(len(eval_calls))
    ])
    df["meta"] = pd.Series([
        eval_calls[i]["meta"] for i in range(len(eval_calls))
    ])

    return df.join(df.meta.apply(lambda m: pd.Series(m))).drop(
        columns=["meta", "output", "metadata"], errors="ignore"
    )


def display_feedback_call(
    record_id: str,
    call: List[Dict[str, Any]],
    feedback_name: str,
    feedback_directions: Dict[str, bool],
):
    """Display feedback call details in a DataFrame. For OTel spans, this function filters and displays EVAL spans only, not EVAL_ROOT spans.

    Args:
        record_id (str): The record ID.
        call (List[Dict[str, Any]]): The feedback call details, including call metadata.
        feedback_name (str): The feedback name.
        feedback_directions (Dict[str, bool]): A dictionary mapping feedback names to their directions. True if higher is better, False otherwise.
    """
    if not call:
        st.warning("No feedback details found.")
        return

    # First, identify and separate EVAL_ROOT and feedback calls (EVAL spans)
    eval_root_calls, eval_calls = _identify_span_types(call)

    # For OTel spans only: filter EVAL_ROOT spans to get most recent ones
    eval_calls = _filter_eval_calls_by_root(eval_root_calls, eval_calls)

    if not eval_calls:
        st.warning("No feedback details found.")
        return

    # Process feedback calls (EVAL spans) for display
    df = _process_eval_calls_for_display(eval_calls)

    # Handle groundedness feedback specially
    if "groundedness" in feedback_name.lower():
        try:
            df = expand_groundedness_df(df)
        except ValueError:
            st.error(
                "Error expanding groundedness DataFrame. "
                "Please ensure the DataFrame is in the correct format."
            )

    if df.empty:
        st.warning("No feedback details found.")
        return

    # Style and display the DataFrame
    style_highlight_fn = partial(
        highlight,
        selected_feedback=feedback_name,
        feedback_directions=feedback_directions,
        default_direction=default_direction,
    )
    styled_df = df.style.apply(
        style_highlight_fn,
        axis=1,
    )

    # Format only numeric columns
    for col in df.select_dtypes(include=["number"]).columns:
        styled_df = styled_df.format({col: "{:.2f}"})

    st.dataframe(styled_df, hide_index=True)


def _filter_duplicate_span_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only show rows from the most recent eval_root_id for duplicate evaluations.

    Groups EVAL_ROOT spans by (args_span_id, args_span_attribute) to identify unique evaluation inputs,
    then keeps only the EVAL_ROOT and EVAL rows belonging to the most recent eval_root_id based on timestamp.

    Args:
        df: DataFrame containing feedback call data with potential duplicates

    Returns:
        DataFrame with duplicate span calls filtered to show only rows from the most recent eval_root_id

    Raises:
        KeyError: If required columns (eval_root_id, timestamp) are missing
        ValueError: If timestamps are invalid (None or not parseable)
    """
    # Early exit if required columns are missing
    required_columns = {"eval_root_id", "timestamp"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Required columns missing: {missing_columns}")

    if df.empty:
        return df

    # Verify timestamps are valid
    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamps: None values found")

    # If we have args_span_id and args_span_attribute columns, do sophisticated deduplication
    if {"args_span_id", "args_span_attribute"}.issubset(df.columns):
        # Create string columns for grouping
        args_span_id_str = (
            df["args_span_id"]
            .astype(str)
            .where(df["args_span_id"].notna(), None)
        )
        args_span_attribute_str = (
            df["args_span_attribute"]
            .astype(str)
            .where(df["args_span_attribute"].notna(), None)
        )

        # Group by the combination of args_span_id and args_span_attribute
        grouped = df.groupby(
            [args_span_id_str, args_span_attribute_str], dropna=False
        )

        # Find the most recent eval_root_id for each group
        most_recent_indices = []
        for _, group_df in grouped:
            if len(group_df) <= 1:
                # Single eval_root_id - keep all rows
                most_recent_indices.extend(group_df.index)
            else:
                # Multiple eval_root_ids - keep only the most recent one
                most_recent_idx = group_df["timestamp"].idxmax()
                most_recent_indices.append(most_recent_idx)

        # Filter to keep only the most recent rows
        filtered_df = df.loc[most_recent_indices].copy()

        # Drop columns that were only needed for filtering, but keep eval_root_id
        return filtered_df.drop(
            columns=["args_span_id", "args_span_attribute", "timestamp"],
            errors="ignore",
        )
    else:
        # If we only have spans with no args_span_id or args_span_attribute,
        # return all spans without filtering
        return df.drop(columns=["timestamp"], errors="ignore")


def _render_feedback_pills(
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    selected_row: Optional[pd.Series] = None,
):
    """Render each feedback as pills.

    Args:
        feedback_col_names (Sequence[str]): The name of the feedback function columns.
        feedback_directions (Dict[str, bool]): A dictionary mapping feedback names to their directions. True if higher is better, False otherwise.
        selected_row (Optional[pd.Series], optional): The selected row (if any). If provided, renders the feedback values. Defaults to None.

    Returns:
        Any: The feedback pills streamlit component.
    """
    if selected_row is not None:
        # Initialize session state for selected feedback if not already set

        def get_icon(feedback_name: str):
            cat = CATEGORY.of_score(
                selected_row[feedback_name],
                higher_is_better=feedback_directions.get(feedback_name, True),
            )
            return cat.icon

        feedback_with_valid_results = sorted([
            fcol
            for fcol in feedback_col_names
            if fcol in selected_row and selected_row[fcol] is not None
        ])

        format_func = (
            lambda fcol: f"{get_icon(fcol)} {fcol} {selected_row[fcol]:.2f}"
        )
    else:
        feedback_with_valid_results = feedback_col_names
        format_func = None

    if len(feedback_with_valid_results) == 0:
        st.warning("No feedback functions found.")
        return

    kwargs = {
        "label": "Feedback Functions (click to learn more)",
        "options": feedback_with_valid_results,
    }
    if format_func:
        kwargs["format_func"] = format_func

    if hasattr(st, "pills"):
        # Use native streamlit pills, released in 1.40.0
        return st.pills(
            **kwargs,
        )
    else:
        return st.selectbox(**kwargs, index=None)


def _render_feedback_call(
    feedback_col: str,
    selected_row: pd.Series,
    feedback_directions: Dict[str, bool],
):
    fcol = feedback_col
    if MULTI_CALL_NAME_DELIMITER in feedback_col:
        fcol = feedback_col.split(MULTI_CALL_NAME_DELIMITER)[0]

    feedback_calls = selected_row[f"{feedback_col}_calls"]
    display_feedback_call(
        selected_row["record_id"],
        feedback_calls,
        fcol,
        feedback_directions=feedback_directions,
    )

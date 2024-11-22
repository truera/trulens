from functools import partial
import pprint as pp
from typing import Any, Dict, List, Optional, Sequence

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


def display_feedback_call(
    record_id: str,
    call: List[Dict[str, Any]],
    feedback_name: str,
    feedback_directions: Dict[str, bool],
):
    """Display the feedback call details in a DataFrame.

    Args:
        record_id (str): The record ID.
        call (List[Dict[str, Any]]): The feedback call details, including call metadata.
        feedback_name (str): The feedback name.
        feedback_directions (Dict[str, bool]): A dictionary mapping feedback names to their directions. True if higher is better, False otherwise.
    """
    if call is not None and len(call) > 0:
        # NOTE(piotrm for garett): converting feedback
        # function inputs to strings here as other
        # structures get rendered as [object Object] in the
        # javascript downstream. If the first input/column
        # is a list, the DataFrame.from_records does create
        # multiple rows, one for each element, but if the
        # second or other column is a list, it will not do
        # this.
        for c in call:
            args: Dict = c["args"]
            for k, v in args.items():
                if not isinstance(v, str):
                    args[k] = pp.pformat(v)

        df = pd.DataFrame.from_records(c["args"] for c in call)

        df["score"] = pd.DataFrame([
            float(call[i]["ret"]) if call[i]["ret"] is not None else -1
            for i in range(len(call))
        ])
        df["meta"] = pd.Series([call[i]["meta"] for i in range(len(call))])
        df = df.join(df.meta.apply(lambda m: pd.Series(m))).drop(columns="meta")

        # note: improve conditional to not rely on the feedback name
        if "groundedness" in feedback_name.lower():
            df = expand_groundedness_df(df)
        if df.empty:
            st.warning("No feedback details found.")
        else:
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
    else:
        st.warning("No feedback details found.")


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

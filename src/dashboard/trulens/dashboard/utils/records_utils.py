import pprint as pp
import re
from typing import Any, Dict, List, Sequence

import pandas as pd
import streamlit as st
from streamlit_pills import pills
from trulens.core.database.base import MULTI_CALL_NAME_DELIMITER
from trulens.dashboard.ux.styles import CATEGORY
from trulens.dashboard.ux.styles import default_direction


def display_feedback_call(
    call: List[Dict[str, Any]],
    feedback_name: str,
    feedback_directions: Dict[str, bool],
):
    def highlight(s):
        if "distance" in feedback_name:
            return [f"background-color: {CATEGORY.UNKNOWN.color}"] * len(s)
        cat = CATEGORY.of_score(
            s.result,
            higher_is_better=feedback_directions.get(
                feedback_name, default_direction
            )
            == default_direction,
        )
        return [f"background-color: {cat.color}"] * len(s)

    def highlight_groundedness(s):
        if "distance" in feedback_name:
            return [f"background-color: {CATEGORY.UNKNOWN.color}"] * len(s)
        cat = CATEGORY.of_score(
            s.Score,
            higher_is_better=feedback_directions.get(
                feedback_name, default_direction
            )
            == default_direction,
        )
        return [f"background-color: {cat.color}"] * len(s)

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

        df["result"] = pd.DataFrame([
            float(call[i]["ret"]) if call[i]["ret"] is not None else -1
            for i in range(len(call))
        ])
        df["meta"] = pd.Series([call[i]["meta"] for i in range(len(call))])
        df = df.join(df.meta.apply(lambda m: pd.Series(m))).drop(columns="meta")

        # note: improve conditional to not rely on the feedback name
        if "groundedness" in feedback_name.lower():
            try:
                # Split the reasons value into separate rows and columns
                reasons = df["reasons"].iloc[0]
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
                        supporting_evidence = statement.split(
                            "Supporting Evidence: "
                        )[1].split("Score: ")[0]
                        score_pattern = re.compile(r"([0-9]+)(?=\D*$)")
                        score_split = statement.split("Score: ")[1]
                        score_match = score_pattern.search(score_split)
                        if score_match:
                            score = float(score_match.group(1)) / 10
                    except Exception:
                        pass
                    data.append({
                        "Statement": criteria,
                        "Supporting Evidence from Source": supporting_evidence,
                        "Score": score,
                    })
                reasons_df = pd.DataFrame(data)
                # Combine the original feedback data with the expanded reasons
                df_expanded = pd.concat(
                    [
                        df.reset_index(drop=True),
                        reasons_df.reset_index(drop=True),
                    ],
                    axis=1,
                )
                st.dataframe(
                    df_expanded.style.apply(
                        highlight_groundedness, axis=1
                    ).format("{:.2f}", subset=["Score"]),
                    hide_index=True,
                    column_order=[
                        "Statement",
                        "Supporting Evidence from Source",
                        "Score",
                    ],
                )
                return
            except Exception:
                pass

        st.dataframe(
            df.style.apply(highlight, axis=1),
            hide_index=True,
        )
    else:
        st.text("No feedback details.")


def _render_feedback_pills(
    feedback_col_names: Sequence[str],
    selected_row: pd.Series,
    feedback_directions: Dict[str, bool],
):
    if len(feedback_col_names) == 0:
        st.write("No feedback details")
        return
    feedback_with_valid_results = sorted(
        list(
            filter(
                lambda fcol: selected_row[fcol] is not None, feedback_col_names
            )
        )
    )

    def get_icon(feedback_name: str):
        cat = CATEGORY.of_score(
            selected_row[feedback_name],
            higher_is_better=feedback_directions.get(feedback_name, True),
        )
        return cat.icon

    icons = list(map(lambda fcol: get_icon(fcol), feedback_with_valid_results))

    if len(feedback_with_valid_results) == 0:
        st.write("No feedback functions found.")
        return

    return pills(
        "Feedback functions (click on a pill to learn more)",
        feedback_with_valid_results,
        index=None,
        format_func=lambda fcol: f"{fcol} {selected_row[fcol]:.4f}",
        icons=icons,
    )


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
        feedback_calls,
        fcol,
        feedback_directions=feedback_directions,
    )

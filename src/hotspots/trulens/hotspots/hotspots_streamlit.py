from dataclasses import replace
from typing import Optional

import pandas as pd
import streamlit as st
from trulens.hotspots.hotspots import HotspotsConfig
from trulens.hotspots.hotspots import get_skipped_columns
from trulens.hotspots.hotspots import hotspots
from trulens.hotspots.hotspots import hotspots_dict_to_df


def _does_streamlit_handle_row_selection() -> bool:
    current_version = st.__version__
    ver_parts = list(map(int, current_version.split(".")))
    return ver_parts[0] >= 2 or ver_parts[0] == 1 and ver_parts[1] >= 35


def _print_header(level: int, h: str) -> None:
    st.markdown(level * "#" + " " + h)


def hotspots_in_streamlit_with_config(
    hotspots_config: HotspotsConfig,
    df: pd.DataFrame,
    header_level: int = 2,
    show_per_sample_results: bool = True,
    select_hotspot_column: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Shows hotspots as a part of a streamlit app for a given config.

    Parameters:
    config (HotspotConfig): config specifying which column contains the evaluation score and other options
    df (DataFrame) - Pandas data frame with per-item evaluation scores
    header_level - the header level for the main data frame (+1 for the hotspots data frame)
    show_per_sample_results - whether to show the main, per-sample data frame
    select_hotspot_column - whether to show the checkboxes with the list of hotspot columns

    Returns:
      - the subset of samples with selected hotspot(s), or None if none selected
    """

    score_column = hotspots_config.score_column
    columns_shown = [score_column] + [
        c for c in df.columns if c != score_column
    ]
    df_to_be_shown = df[columns_shown]

    if show_per_sample_results:
        _print_header(header_level, "Per-sample results")

        styled_df = df_to_be_shown.style.applymap(
            lambda _: "background-color: lightgreen", subset=[score_column]
        )

        st.write(styled_df)

    if select_hotspot_column:
        _print_header(header_level + 1, "Columns where to look for hotspots")

        st.write(
            "You should avoid columns correlating with the main score in a trivial way"
        )

        skipped_columns = get_skipped_columns(hotspots_config, df.columns)

        columns = [c for c in df.columns if c != hotspots_config.score_column]

        wanted_states = {}

        num_of_cb_columns = 6
        cb_columns = st.columns(num_of_cb_columns)
        for i, col in enumerate(columns):
            with cb_columns[i % num_of_cb_columns]:
                wanted_states[col] = st.checkbox(
                    label=col, value=col not in skipped_columns
                )

        # config with skipped column overridden by checkboxes
        new_config = replace(
            hotspots_config,
            skip_columns=[c for c, sel in wanted_states.items() if not sel],
            more_skipped_columns=None,
        )
    else:
        new_config = hotspots_config

    out_df, avg_score, htss = hotspots(new_config, df)

    hotspots_df = hotspots_dict_to_df(out_df, avg_score, htss)

    column_config = {
        "#": st.column_config.NumberColumn(
            "#",
            help="number of occurences",
        ),
        "avg. score": st.column_config.NumberColumn(
            "avg. score",
            help="average score",
        ),
        "deterioration": st.column_config.NumberColumn(
            "deterioration",
            help="the delta between the average score for samples containing the feature and the average score for the rest of samples",
        ),
        "opportunity": st.column_config.NumberColumn(
            "opportunity",
            help="how much the average _total_ score would improve if we have somehow fixed the problem with the hotspot",
        ),
        "p-value": st.column_config.NumberColumn(
            "p-value",
            help="how likely, assuming the effect is due to chance",
        ),
    }

    extra_opts = {"column_config": column_config}

    _print_header(header_level + 1, "Hotspots")

    samples_selected = None

    if _does_streamlit_handle_row_selection():
        st.write(
            "You can select hotspot(s) to see rows with them (but be patient!)"
        )

        event = st.dataframe(
            hotspots_df,
            on_select="rerun",
            selection_mode="multi-row",
            **extra_opts,
        )

        selected_hotspots_idxs = event.selection.rows

        if selected_hotspots_idxs:
            selected_row_lists = [
                htss[idx][2] for idx in selected_hotspots_idxs
            ]
            rows_idxs = sorted(set(sum(selected_row_lists, [])))

            _print_header(header_level + 1, "Rows with selected hotspots")

            samples_selected = df_to_be_shown.iloc[rows_idxs]

            st.write(samples_selected)
    else:
        st.dataframe(hotspots_df, **extra_opts)

    return samples_selected


def hotspots_in_streamlit(df: pd.DataFrame) -> None:
    """
    Shows hotspots as a part of a streamlit app.

    Parameters:
    df (DataFrame) - Pandas data frame with per-item evaluation scores
    """
    columns = df.columns.tolist()

    selected_column = st.selectbox(
        "Column with the evaluation score", columns, index=None
    )

    lower_is_better = st.checkbox(
        "The-lower-the-better metric (e.g. MAE, RMSE, WER)", value=False
    )

    if selected_column:
        hotspots_config = HotspotsConfig(
            score_column=selected_column,
            higher_is_better=not lower_is_better,
        )

        hotspots_in_streamlit_with_config(hotspots_config, df)


def sample_main() -> None:
    st.set_page_config(layout="wide")

    st.markdown("# Sample Streamlit application for TruLens Hotspots")
    st.write("Upload a CSV file (can be compressed) with evaluation results")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "gz"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".gz"):
            df = pd.read_csv(uploaded_file, compression="gzip")
        else:
            df = pd.read_csv(uploaded_file)
        hotspots_in_streamlit(df)


if __name__ == "__main__":
    import sys

    from streamlit import runtime
    from streamlit.web import cli as stcli

    if runtime.exists():
        sample_main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

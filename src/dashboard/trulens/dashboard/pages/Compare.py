"""
Do not edit this code directly on Snowflake Streamlit.
Use https://github.com/snowflakedb/aiml-cortex-doc-summit-demo
eval > streamlit_app > main.py
as source of truth
"""

import json

import pandas as pd
import streamlit as st
from trulens.core import Tru

tru = Tru()

st.set_page_config(layout="wide")

if "compute_button" not in st.session_state:
    st.session_state["compute_button"] = False


def set_compute_button():
    st.session_state["compute_button"] = True


def sxs_ab_columns(label: str):
    col_a, col_b = st.columns(2)
    col_a.markdown(f"##### {label} (A)")
    col_b.markdown(f"##### {label} (B)")
    return col_a, col_b


def sxs_ab_columns_with_score(score_label: str, score_a: float, score_b: float):
    col_a, col_b = st.columns(2)
    col_a.markdown(f"{score_label}: {score_a}")
    col_b.markdown(f"{score_label}: {score_b}")
    return col_a, col_b


def render_sxs(label: str, text_a: str, text_b: str, format: str = ""):
    col_a, col_b = sxs_ab_columns(label)
    if format == "rag_context":
        render_rag_context(text_a, col_a, "a")
        render_rag_context(text_b, col_b, "b")
    elif format == "citation" or format == "debug_signals":
        col_a.json(text_a)
        col_b.json(text_b)
    else:
        col_a.markdown(text_a)
        col_b.markdown(text_b)


def render_sxs_records_link(display_a, display_b):
    col_a, col_b = sxs_ab_columns("Record Span Details")
    if col_a.button("View in Records", key="view_in_records_a"):
        st.session_state["record_id"] = display_a.record_id.values[0]
        st.switch_page("Records")
    if col_b.button("View in Records", key="view_in_records_b"):
        st.session_state["record_id"] = display_a.record_id.values[0]
        st.switch_page("Records")


def render_sxs_with_scores(
    label: str,
    text_a: str,
    text_b: str,
    score_labels: list = [],
    scores_a: list = [],
    scores_b: list = [],
    format: str = "",
):
    col_a, col_b = sxs_ab_columns(label)

    # Display scores if provided
    if score_labels and scores_a and scores_b:
        for score_label, score_a, score_b in zip(
            score_labels, scores_a, scores_b
        ):
            col_a.markdown(f"**{score_label}**: {score_a}")
            col_b.markdown(f"**{score_label}**: {score_b}")

    if format == "rag_context":
        render_rag_context(text_a, col_a, "a")
        render_rag_context(text_b, col_b, "b")
    elif format == "citation" or format == "debug_signals":
        col_a.json(text_a)
        col_b.json(text_b)
    else:
        col_a.markdown(text_a)
        col_b.markdown(text_b)


def render_rag_context(rag_obj: str, comp, comp_key: str):
    for index, r in enumerate(rag_obj):
        expander = comp.expander(f"rank: {r['index']}")
        raw_view = expander.toggle(
            "display raw", key=f"{comp_key}-{index}-raw-toggle"
        )
        if raw_view:
            expander.write(json.dumps(r, indent=4))
        else:
            expander.text_area(
                "excerpt",
                r["excerpt"],
                height=300,
                key=f"{comp_key}-{index}-excerpt",
            )
            expander.caption(f"source: {r['source']}")
            expander.markdown(f"[presigned_url]({r['presigned_url']})")


def render_citation_stats(match: int, duplicated: int, leak: int, comp):
    cs_col1, cs_col2, cs_col3 = comp.columns(3)
    cs_col1.metric("matches", match)
    cs_col2.metric("duplicated", duplicated)
    cs_col3.metric(
        "leak",
        leak,
        help="generated citation even when system prompt asked not to",
    )


def render_citation_stats_sxs(
    label: str,
    match_a: int,
    duplicated_a: int,
    leak_a: int,
    match_b: int,
    duplicated_b: int,
    leak_b: int,
):
    col_a, col_b = sxs_ab_columns(label)
    render_citation_stats(match_a, duplicated_a, leak_a, col_a)
    render_citation_stats(match_b, duplicated_b, leak_b, col_b)


def render_scores_sxs(label: str, df_a, df_b):
    columns = {
        "is_match": "Exact Match",
        "anls": "ANLS",
        "accuracy_llm": "LLM Accuracy",
        "retrieval_anls": "ANLS",
        "retrieval_ndcg@1": "NDCG@1",
        "retrieval_hit@1": "Hit@1",
        "retrieval_ndcg@3": "NDCG@3",
        "retrieval_hit@3": "Hit@3",
        "retrieval_ndcg@5": "NDCG@5",
        "retrieval_hit@5": "Hit@5",
        "retrieval_ndcg@10": "NDCG@10",
        "retrieval_hit@10": "Hit@10",
    }

    scores_a = {}
    scores_b = {}

    for key, column in columns.items():
        if column in df_a.columns:
            scores_a[key] = df_a[column].values[0]
        if column in df_b.columns:
            scores_b[key] = df_b[column].values[0]

    combined_scores = {
        "Metric": list(columns.keys()),
        "Experiment (A)": [scores_a.get(key, None) for key in columns.keys()],
        "Experiment (B)": [scores_b.get(key, None) for key in columns.keys()],
    }

    combined_scores_df = pd.DataFrame(combined_scores)
    combined_scores_df.set_index("Metric", inplace=True)
    st.table(combined_scores_df)


def render_summary(df, comp):
    exact_match = (df["Exact Match"].mean()) * 100
    anls_avg = df["ANLS"].mean() * 100
    accuracy_llm_avg = (
        (df[df["LLM Accuracy"] >= 0]["LLM Accuracy"].mean() / 2.0) * 100
        if not (df["LLM Accuracy"] == -1).all()
        else -1
    )
    accuracy_llm_judged_total = len(df[df["LLM Accuracy"] >= 0])

    # df["GOLDEN"] = df["GOLDEN"].apply(json.loads)
    # df["RAG_CONTEXT"] = df["RAG_CONTEXT"].apply(json.loads)
    # abstain_score = 1 - df[df["GOLDEN"].apply(len) == 0]["RAG_CONTEXT"].apply(
    #    len
    # ).mean() / max(df["RAG_CONTEXT"].apply(len))
    citation_match_score = (
        df[df["Citation Match"] >= 0]["Citation Match"].sum()
        / (len(df[df["Citation Match"] >= 0]) * 2)
    ) * 100
    citation_duplicate_score = (
        df[df["Citation Duplicated"] >= 0]["Citation Duplicated"].mean()
    ) * 100
    # calculate extra citation
    citation_leak_score = (
        df[df["Extra Citation"] >= 0]["Extra Citation"].mean()
    ) * 100

    comp.text(f"Average EM score: {exact_match:.2f}%")
    comp.text(f"Average ANLS score: {anls_avg:.4f}%")
    comp.text(f"Average LLM accuracy score: {accuracy_llm_avg:.4f}%")
    comp.text(
        f"Total LLM judged: {accuracy_llm_judged_total} out of {len(df)} rows."
    )
    # comp.text(f"Average abstain score: {abstain_score:.4f}")
    # for k in [1, 3, 5, 10]:
    #    col_key_ndcg = f"NDCG@{k}"
    #    col_key_hit = f"Hit{k}"
    #    comp.text(
    #        f"Average retrieval NDCG@{k}: {df[df['GOLDEN'].apply(len) > 0][col_key_ndcg].mean():.4f}"
    #    )
    #    comp.text(
    #        f"Average retrieval Hit@{k}: {df[df['GOLDEN'].apply(len) > 0][col_key_hit].mean():.4f}"
    #    )

    comp.markdown("###### Citation Stats")
    comp.text(f"Percentage of citation matches: {citation_match_score:.2f}%")
    comp.text(
        f"With applicable entries for citation matches : {df[df['Citation Match'] >= 0].shape[0]} out of {len(df)} rows."
    )
    comp.text(
        f"Percentage of citation duplicated: {citation_duplicate_score:.2f}%"
    )
    comp.text(
        f"With applicable entries for citation duplicated : {df[df['Citation Duplicated'] >= 0].shape[0]} out of {len(df)} rows."
    )
    comp.text(f"Percentage of extra citations: {citation_leak_score:.2f}%")
    comp.text(
        f"With applicable entries for extra citations: {df[df['Citation Extra'] >= 0].shape[0]} out of {len(df)} rows."
    )


def render_summary_sxs(label: str, df_a, df_b):
    col_a, col_b = sxs_ab_columns(label)
    render_summary(df_a, col_a)
    render_summary(df_b, col_b)


st.title("Compare App Versions")

apps = tru.get_apps()
apps = list(tru.get_apps())
app_ids = [app["app_id"] for app in apps]


def group_experiments(experiments):
    groups = {}

    for s in experiments:
        if "/" in s:
            key = s.rsplit("/", 1)[0]
            value = s.rsplit("/", 1)[1]
        else:
            key = "one_off_experiments"
            value = s

        if key not in groups:
            groups[key] = []

        groups[key].append(value)

    return groups


experiment_groups = group_experiments(app_ids)

st.write("## Select Experiments")


def create_experiment_group_widget():
    "Select an experiment group"
    experiment_group = "all experiments"

    experiment_group = st.selectbox(
        "Experiment Group",
        options=list(["all experiments"] + list(experiment_groups.keys())),
    )
    # Determine the app_ids to display based on the selected experiment group
    if experiment_group == "all experiments":
        selected_app_ids = app_ids
    else:
        selected_app_ids = experiment_groups[experiment_group]
    return selected_app_ids, experiment_group


selected_app_ids, experiment_group = create_experiment_group_widget()


input_col_a, input_col_b = st.columns(2)


if "app_ids" in st.session_state and len(st.session_state["app_ids"]) == 2:
    exp_a, exp_b = st.session_state["app_ids"]
    if exp_a in selected_app_ids and exp_b in selected_app_ids:
        exp_a_idx = selected_app_ids.index(exp_a)
        exp_b_idx = selected_app_ids.index(exp_b)
    else:
        exp_a_idx = 0
        exp_b_idx = 1

experiment_a = input_col_a.selectbox(
    "Experiment A", options=selected_app_ids, index=exp_a_idx
)
experiment_b = input_col_b.selectbox(
    "Experiment B", options=selected_app_ids, index=exp_b_idx
)

if experiment_group == "all experiments":
    app_id_a = experiment_a
    app_id_b = experiment_b
else:
    app_id_a = f"{experiment_group}/{experiment_a}"
    app_id_b = f"{experiment_group}/{experiment_b}"

if (
    st.button("Compare", type="primary", on_click=set_compute_button)
    or st.session_state["compute_button"]
):
    table_a_records, table_a_feedback = tru.get_records_and_feedback(
        app_ids=[app_id_a]
    )
    table_b_records, table_b_feedback = tru.get_records_and_feedback(
        app_ids=[app_id_b]
    )

    if len(table_a_records) == 0 or len(table_b_records) == 0:
        st.error("No records found for the selected apps")
        st.stop()

    def get_columns(records, feedback):
        base_columns = [
            "record_id",
            "app_id",
            "record_json",
            "input",
            "output",
            "latency",
        ]
        optional_columns = ["Exact Match_calls", "Citation Match_calls"]

        # Add optional columns if they exist in the records
        for col in optional_columns:
            if col in records.columns:
                base_columns.append(col)

        return base_columns + feedback

    # Get columns for table_a and table_b
    table_a_columns = get_columns(table_a_records, table_a_feedback)
    table_b_columns = get_columns(table_b_records, table_b_feedback)

    # Create tables with the selected columns
    table_a = table_a_records[table_a_columns]
    table_b = table_b_records[table_b_columns]

    # Filter the tables based on the selected columns
    table_a_intersection = table_a[table_a["input"].isin(table_b["input"])]
    table_b_intersection = table_b[table_b["input"].isin(table_a["input"])]

    # Filter the tables based on the SQL where clause - # temporarily skipping this step
    table_a_filtered, table_b_filtered = (
        table_a_intersection,
        table_b_intersection,
    )

    st.markdown(
        f"#### There are {len(table_a_filtered)} queries match your filters"
    )
    if len(table_a_filtered) > 0:
        # using filtered_a instead of qualify_queries_df in order to
        # show original query num in output
        st.dataframe(table_a_filtered["input"])

        query_option = st.selectbox(
            "Select a query to see SxS result",
            options=table_a_filtered["input"],
        )

        query_index = table_a_filtered["input"].tolist().index(query_option)

        display_a = table_a_filtered[table_a_filtered["input"] == query_option]
        display_b = table_b_filtered[table_b_filtered["input"] == query_option]

        st.divider()
        render_sxs_records_link(display_a, display_b)

        st.divider()
        if "Exact Match_calls" in table_a_columns:
            st.markdown("##### :orange[Golden Response]")
            st.write(
                display_a["Exact Match_calls"][query_index][0]["args"][
                    "ground_truth_response"
                ]
            )
        st.write("\n")
        score_labels = []
        scores_a = []
        scores_b = []

        if "Exact Match" in table_a_columns:
            score_labels.append("Exact Match")
            scores_a.append(display_a["Exact Match"].values[0])
            scores_b.append(display_b["Exact Match"].values[0])

        if "LLM Accuracy" in table_a_columns:
            score_labels.append("LLM Accuracy")
            scores_a.append(display_a["LLM Accuracy"].values[0])
            scores_b.append(display_b["LLM Accuracy"].values[0])

        render_sxs_with_scores(
            "LLM Answer",
            display_a["output"].values[0],
            display_b["output"].values[0],
            score_labels,
            scores_a,
            scores_b,
        )

        if (
            "adjusted_answer"
            in json.loads(display_a["record_json"].values[0])["calls"][0][
                "rets"
            ]
        ):
            render_sxs(
                "Adjusted LLM Answer",
                json.loads(display_a["record_json"].values[0])["calls"][0][
                    "rets"
                ]["adjusted_answer"],
                json.loads(display_b["record_json"].values[0])["calls"][0][
                    "rets"
                ]["adjusted_answer"],
            )
        st.divider()

        if "Citation Match_calls" in table_a_columns:
            st.markdown("##### :orange[Golden Citation(s)]")
            st.json(
                display_a["Citation Match_calls"].values[0][0]["args"][
                    "citations"
                ]
            )
        if (
            "citations"
            in json.loads(display_a["record_json"].values[0])["calls"][0][
                "rets"
            ]
        ):
            st.write("\n")
            # Collect score labels and values for columns starting with "Citations"
            score_labels = []
            scores_a = []
            scores_b = []

            for column in table_a_columns:
                if column.startswith("Citation") and not column.endswith(
                    "_calls"
                ):
                    score_labels.append(column)
                    scores_a.append(display_a[column].values[0])
                    scores_b.append(display_b[column].values[0])

            # Render the side-by-side comparison with scores
            render_sxs_with_scores(
                "Citations",
                json.loads(display_a["record_json"].values[0])["calls"][0][
                    "rets"
                ]["citations"],
                json.loads(display_b["record_json"].values[0])["calls"][0][
                    "rets"
                ]["citations"],
                score_labels,
                scores_a,
                scores_b,
                format="citation",
            )
        # Check if the necessary keys exist in display_a and display_b for render_citation_stats_sxs
        if all(
            key in display_a
            for key in [
                "Citation Match",
                "Citation Duplicated",
                "Extra Citation",
            ]
        ) and all(
            key in display_b
            for key in [
                "Citation Match",
                "Citaiton Duplicated",
                "Extra Citation Match",
            ]
        ):
            render_citation_stats_sxs(
                "Citation Stats",
                json.loads(display_a["Citation Match"].values[0]),
                json.loads(display_a["Citation Duplicated"].values[0]),
                json.loads(display_a["Extra Citation"].values[0]),
                json.loads(display_b["Citation Match"].values[0]),
                json.loads(display_b["Citaiton Duplicated"].values[0]),
                json.loads(display_b["Extra Citation Match"].values[0]),
            )

        # Check if the necessary keys exist in display_a and display_b for render_sxs
        if (
            "debug_signals"
            in json.loads(display_a["record_json"].values[0])["calls"][0][
                "rets"
            ]
            and "debug_signals"
            in json.loads(display_b["record_json"].values[0])["calls"][0][
                "rets"
            ]
        ):
            debug_signals_a = json.loads(display_a["record_json"].values[0])[
                "calls"
            ][0]["rets"]["debug_signals"]
            debug_signals_b = json.loads(display_b["record_json"].values[0])[
                "calls"
            ][0]["rets"]["debug_signals"]
            render_sxs(
                "Debug Signals",
                debug_signals_a,
                debug_signals_b,
                format="debug_signals",
            )

        st.divider()

        st.markdown("##### :orange[Golden Excerpts]")
        st.write("TODO: display golden excerpts")
        # st.json(display_a["GOLDEN"].values[0])
        if (
            "rag_context"
            in json.loads(display_a["record_json"].values[0])["calls"][0][
                "rets"
            ]
        ):
            st.write("\n")
            # Define the relevant columns
            relevant_columns = ["ANLS", "Hit@1", "Hit@3", "NDCG@1", "NDCG@3"]

            # Collect score labels and values for the relevant columns
            score_labels = []
            scores_a = []
            scores_b = []

            for column in relevant_columns:
                if column in table_a_columns:
                    score_labels.append(column)
                    scores_a.append(display_a[column].values[0])
                    scores_b.append(display_b[column].values[0])

            # Render the side-by-side comparison with scores
            render_sxs_with_scores(
                "Rag Context",
                json.loads(display_a["record_json"].values[0])["calls"][0][
                    "rets"
                ]["rag_context"],
                json.loads(display_b["record_json"].values[0])["calls"][0][
                    "rets"
                ]["rag_context"],
                score_labels,
                scores_a,
                scores_b,
                format="rag_context",
            )

        st.divider()
        st.markdown("#### Score Summary")
        render_scores_sxs("Scores", display_a, display_b)

        st.divider()
        st.markdown("#### Experiment Summary for all queries")
        render_scores_sxs("Scores", table_a, table_b)

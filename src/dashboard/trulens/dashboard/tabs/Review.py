"""Human review queues.

Creates queues from the records already loaded in the dashboard, pulls one
item at a time, renders the trace alongside the fixed review fields, and
exports completed reviews. All state lives in the configured database and all
work runs in this process: there is no background worker.
"""

from typing import List, Optional

import pandas as pd
import streamlit as st
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.review import ReviewTargets
from trulens.core.schema import review as review_schema
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.components.record_viewer_otel import record_viewer_otel
from trulens.dashboard.constants import REVIEW_PAGE_NAME as page_name
from trulens.dashboard.utils.dashboard_utils import _get_event_otel_spans
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import get_session
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.streamlit_compat import st_columns

PRESET_LOW_SCORE = "Low evaluation score"
PRESET_HIGH_LATENCY = "High latency"
PRESET_HIGH_COST = "High cost"
PRESETS = (PRESET_LOW_SCORE, PRESET_HIGH_LATENCY, PRESET_HIGH_COST)

SELECTED_QUEUE_KEY = f"{page_name}.selected_queue"
CLAIMED_ITEM_KEY = f"{page_name}.claimed_item"
STAGED_TARGETS_KEY = f"{page_name}.staged_targets"


def _feedback_columns(records_df: pd.DataFrame) -> List[str]:
    """Metric columns in a records dataframe.

    A metric is a column that has a matching direction column, which is how
    `get_records_and_feedback` reports evaluation results.
    """

    return sorted(
        column
        for column in records_df.columns
        if f"{column} direction" in records_df.columns
    )


def _build_predicate(
    records_df: pd.DataFrame,
    preset: str,
    metric: Optional[str],
    threshold: float,
    use_top_n: bool,
    top_n: int,
    currency: Optional[str],
):
    """Turn the panel's choices into a selection predicate."""

    if preset == PRESET_LOW_SCORE:
        if not metric:
            return None
        if use_top_n:
            return ReviewTargets.worst_score(metric, top_n=top_n)
        return ReviewTargets.low_score(metric, below=threshold)

    if preset == PRESET_HIGH_LATENCY:
        if use_top_n:
            return ReviewTargets.slowest(top_n=top_n)
        return ReviewTargets.high_latency(above_seconds=threshold)

    if preset == PRESET_HIGH_COST:
        if not currency:
            return None
        if use_top_n:
            return ReviewTargets.most_expensive(top_n=top_n, currency=currency)
        return ReviewTargets.high_cost(above=threshold, currency=currency)

    return None


def render_queue_creation(records_df: pd.DataFrame):
    """Panel for building a queue from the loaded records."""

    st.subheader("Create a queue")

    if records_df is None or records_df.empty:
        st.info("No records are loaded for this app yet.")
        return

    with st.form(f"{page_name}.create_queue", border=True):
        name = st.text_input("Queue name", placeholder="low-groundedness")
        instructions = st.text_area(
            "Instructions for reviewers",
            placeholder="Review low-groundedness support responses.",
        )

        preset = st.radio("Preset", PRESETS, horizontal=True)

        cols = st_columns(3)
        with cols[0]:
            metrics = _feedback_columns(records_df)
            metric = (
                st.selectbox("Metric", metrics)
                if preset == PRESET_LOW_SCORE and metrics
                else None
            )
            if preset == PRESET_LOW_SCORE and not metrics:
                st.caption("No evaluation metrics on these records.")

        with cols[1]:
            use_top_n = st.toggle("Use top-N instead of a threshold")
            top_n = st.number_input(
                "Top N", min_value=1, value=20, step=1, disabled=not use_top_n
            )

        with cols[2]:
            threshold = st.number_input(
                "Threshold",
                value=0.5 if preset == PRESET_LOW_SCORE else 8.0,
                step=0.1,
                disabled=use_top_n,
                help=(
                    "Scores below this, latency above this many seconds, or "
                    "cost above this amount."
                ),
            )
            currency = None
            if preset == PRESET_HIGH_COST:
                currencies = sorted({
                    str(c)
                    for c in records_df.get(
                        "cost_currency", pd.Series(dtype=object)
                    )
                    if c is not None and str(c) != "nan"
                })
                # Costs are never compared across currencies, so one has to be
                # picked rather than defaulted.
                currency = st.selectbox("Currency", currencies or ["USD"])

        limit = st.number_input(
            "Maximum items", min_value=1, value=100, step=10
        )

        previewed = st.form_submit_button("Preview matching records")

    if previewed:
        predicate = _build_predicate(
            records_df, preset, metric, threshold, use_top_n, top_n, currency
        )
        if predicate is None:
            st.warning("This preset needs a metric or currency to be chosen.")
            return

        try:
            targets = ReviewTargets.from_records(
                records_df,
                where=predicate,
                order_by="severity",
                limit=int(limit),
            )
        except ValueError as e:
            st.error(str(e))
            return

        st.session_state[STAGED_TARGETS_KEY] = {
            "name": name,
            "instructions": instructions,
            "targets": targets,
        }

    staged = st.session_state.get(STAGED_TARGETS_KEY)
    if not staged:
        return

    targets = staged["targets"]
    if not targets:
        st.warning("No records match. Try a different threshold.")
        return

    st.caption(f"{len(targets)} matching record(s), worst first.")
    st.dataframe(
        pd.DataFrame([
            {
                "record_id": t.target_id,
                "reason": t.selection.selection_reason,
                "priority": round(t.selection.priority, 3),
            }
            for t in targets
        ]),
        hide_index=True,
        use_container_width=True,
    )

    if st.button("Create queue with these records", type="primary"):
        if not staged["name"]:
            st.error("Give the queue a name.")
            return

        session = get_session()
        # The previewed targets are materialized as-is, so the ids that were
        # shown are exactly the ids that end up in the queue.
        queue = session.create_review_queue(
            name=staged["name"],
            instructions=staged["instructions"] or None,
            targets=targets,
        )
        st.session_state[SELECTED_QUEUE_KEY] = queue.review_queue_id
        st.session_state.pop(STAGED_TARGETS_KEY, None)
        st.success(f"Created queue '{queue.name}' with {len(targets)} items.")
        st.rerun()


def _render_progress(progress: dict):
    """Show a queue's state counts."""

    cols = st_columns(5)
    for col, state in zip(
        cols,
        (
            review_schema.ReviewItemState.PENDING,
            review_schema.ReviewItemState.IN_REVIEW,
            review_schema.ReviewItemState.COMPLETED,
            review_schema.ReviewItemState.SKIPPED,
            review_schema.ReviewItemState.UNAVAILABLE,
        ),
    ):
        with col:
            st.metric(
                state.value.replace("_", " ").title(),
                progress.get(state.value, 0),
            )


def _render_target_context(target_id: str) -> bool:
    """Render the trace behind an item. Returns whether it could be loaded."""

    records_df, _ = get_records_and_feedback(record_ids=[target_id])

    if records_df is None or records_df.empty:
        st.warning(
            "The source trace for this item is no longer available. You can "
            "skip it or mark it unavailable."
        )
        return False

    row = records_df.iloc[0]

    with st.expander("Input and output", expanded=True):
        st.markdown("**Input**")
        st.write(row.get("input"))
        st.markdown("**Output**")
        st.write(row.get("output"))

    with st.expander("Trace", expanded=False):
        if is_otel_tracing_enabled():
            events = _get_event_otel_spans(record_ids=[target_id])
            if events is not None and not events.empty:
                record_viewer_otel(events, key=f"review_{target_id}")
            else:
                st.caption("No trace data available for this record.")
        else:
            record_viewer(
                row.get("record_json"),
                row.get("app_json"),
                key=f"review_{target_id}",
            )

    return True


def _render_selection_reason(item: review_schema.ReviewItem):
    """Show why this item was queued, as frozen at selection time."""

    if item.selection is None:
        return

    selection = item.selection
    st.info(f"Queued because: {selection.selection_reason}")

    details = {
        "priority": round(selection.priority, 3),
        "metric": selection.metric_name,
        "metric value": selection.metric_value,
        "latency": selection.latency,
        "cost": (
            f"{selection.cost} {selection.cost_currency}"
            if selection.cost is not None
            else None
        ),
        "app": (
            f"{selection.app_name} {selection.app_version}"
            if selection.app_name
            else None
        ),
    }
    st.caption(
        " · ".join(f"{k}: {v}" for k, v in details.items() if v is not None)
    )


def render_review_form(item: review_schema.ReviewItem, available: bool):
    """The fixed review fields."""

    session = get_session()

    with st.form(f"{page_name}.review_form", border=True):
        verdict = st.radio(
            "Verdict",
            [v.value for v in review_schema.Verdict],
            horizontal=True,
            help="Required.",
        )
        cols = st_columns(2)
        with cols[0]:
            set_score = st.toggle("Record a score")
            score = st.slider(
                "Score", 0.0, 1.0, 0.5, 0.05, disabled=not set_score
            )
        with cols[1]:
            failure_type = st.selectbox(
                "Failure type",
                [None] + [f.value for f in review_schema.FailureType],
                format_func=lambda v: "—" if v is None else v,
            )

        corrected_output = st.text_area("Corrected output")
        notes = st.text_area("Notes")
        reviewer = st.text_input(
            "Reviewer",
            help="A label, not an authenticated identity.",
        )

        submitted = st.form_submit_button(
            "Submit review", type="primary", disabled=not available
        )

    if submitted:
        try:
            session.submit_human_review(
                target=item.target,
                verdict=verdict,
                score=score if set_score else None,
                failure_type=failure_type,
                corrected_output=corrected_output or None,
                notes=notes or None,
                reviewer=reviewer or None,
                review_item=item,
            )
        except ValueError as e:
            st.error(str(e))
            return

        st.session_state.pop(CLAIMED_ITEM_KEY, None)
        st.success("Review submitted.")
        st.rerun()


def render_review_panel(review_queue_id: str):
    """Pull an item and review it."""

    session = get_session()
    queue = session.get_review_queue(review_queue_id=review_queue_id)
    if queue is None:
        st.warning("That queue no longer exists.")
        return

    if queue.instructions:
        st.caption(queue.instructions)

    _render_progress(session.get_review_queue_progress(review_queue_id))

    item: Optional[review_schema.ReviewItem] = st.session_state.get(
        CLAIMED_ITEM_KEY
    )

    if item is None or item.review_queue_id != review_queue_id:
        if st.button("Pull next item", type="primary"):
            claimed = session.claim_next_review_item(review_queue_id)
            if claimed is None:
                st.info("Nothing left to review in this queue.")
            else:
                st.session_state[CLAIMED_ITEM_KEY] = claimed
                st.rerun()
        return

    st.divider()
    st.markdown(f"**Reviewing** `{item.target_id}`")
    _render_selection_reason(item)

    available = _render_target_context(item.target_id)

    render_review_form(item, available=available)

    cols = st_columns(3)
    with cols[0]:
        if st.button("Skip"):
            session.skip_review_item(item)
            st.session_state.pop(CLAIMED_ITEM_KEY, None)
            st.rerun()
    with cols[1]:
        if st.button("Release"):
            session.release_review_item(item)
            st.session_state.pop(CLAIMED_ITEM_KEY, None)
            st.rerun()
    with cols[2]:
        if st.button("Mark unavailable", disabled=available):
            session.mark_review_item_unavailable(item)
            st.session_state.pop(CLAIMED_ITEM_KEY, None)
            st.rerun()


def render_export(review_queue_id: Optional[str]):
    """Download completed reviews from this process."""

    session = get_session()
    reviews = session.get_human_reviews(review_queue_id=review_queue_id)

    st.subheader("Completed reviews")
    if reviews.empty:
        st.caption("No reviews submitted yet.")
        return

    st.dataframe(reviews, hide_index=True, use_container_width=True)

    cols = st_columns(2)
    with cols[0]:
        st.download_button(
            "Download CSV",
            reviews.to_csv(index=False),
            file_name="human_reviews.csv",
            mime="text/csv",
        )
    with cols[1]:
        st.download_button(
            "Download JSON",
            reviews.to_json(orient="records"),
            file_name="human_reviews.json",
            mime="application/json",
        )


def render_review_page(app_name: Optional[str]):
    """The Review page."""

    session = get_session()
    queues = session.get_review_queues()

    tab_review, tab_create, tab_export = st.tabs([
        "Review",
        "Create queue",
        "Export",
    ])

    selected: Optional[str] = None
    if not queues.empty:
        names = list(queues["name"])
        ids = list(queues["review_queue_id"])
        current = st.session_state.get(SELECTED_QUEUE_KEY)
        index = ids.index(current) if current in ids else 0
        with tab_review:
            choice = st.selectbox(
                "Queue",
                range(len(ids)),
                index=index,
                format_func=lambda i: names[i],
            )
            selected = ids[choice]
            st.session_state[SELECTED_QUEUE_KEY] = selected
            render_review_panel(selected)
    else:
        with tab_review:
            st.info("No review queues yet. Create one to get started.")

    with tab_create:
        records_df = None
        if app_name:
            records_df, _ = get_records_and_feedback(app_name=app_name)
        render_queue_creation(records_df)

    with tab_export:
        render_export(selected)


def review_main():
    set_page_config(page_title=page_name)
    app_name = render_sidebar()
    render_review_page(app_name)


if __name__ == "__main__":
    review_main()

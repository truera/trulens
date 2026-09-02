"""Shared controls for adding records to a human review queue.

Used by the Records and Compare pages so that a record being looked at can be
sent to a queue without leaving the page.
"""

from typing import List, Optional, Sequence

import streamlit as st
from trulens.core.schema import review as review_schema
from trulens.dashboard.utils.dashboard_utils import get_session

NEW_QUEUE_LABEL = "➕ New queue…"


def render_add_to_queue(
    record_ids: Sequence[str],
    key: str,
    label: str = "Add to review queue",
):
    """Render a control that puts `record_ids` into a review queue.

    Targets added this way carry a `manually selected` reason rather than a
    metric threshold, so a reviewer can still see why the item is in front of
    them.

    Args:
        record_ids: Records to queue.
        key: Unique key prefix for the widgets.
        label: Label for the expander.
    """

    record_ids = [str(r) for r in record_ids if r]
    if not record_ids:
        return

    session = get_session()

    with st.expander(label, expanded=False):
        queues = session.get_review_queues()
        names: List[str] = list(queues["name"]) if not queues.empty else []
        ids: List[str] = (
            list(queues["review_queue_id"]) if not queues.empty else []
        )

        options = names + [NEW_QUEUE_LABEL]
        choice = st.selectbox("Queue", options, key=f"{key}.queue_choice")

        new_name: Optional[str] = None
        if choice == NEW_QUEUE_LABEL:
            new_name = st.text_input("New queue name", key=f"{key}.new_name")

        if not st.button(f"Add {len(record_ids)} record(s)", key=f"{key}.add"):
            return

        if choice == NEW_QUEUE_LABEL:
            if not new_name:
                st.error("Give the queue a name.")
                return
            queue = session.create_review_queue(name=new_name)
            review_queue_id = queue.review_queue_id
        else:
            review_queue_id = ids[names.index(choice)]

        targets = [
            review_schema.ReviewTarget(
                target_id=record_id,
                selection=review_schema.SelectionSnapshot(
                    selection_reason="manually selected",
                    priority=0.0,
                ),
            )
            for record_id in record_ids
        ]

        # Adding a record that is already queued is a no-op, so pressing this
        # twice cannot create duplicate work.
        session.add_review_targets(review_queue_id, targets)
        st.success(f"Added {len(targets)} record(s) to the queue.")

import asyncio
from typing import Optional

import streamlit as st
from trulens.core import TruSession
from trulens.core.schema import app as mod_app_schema
from trulens.core.schema import record as mod_record_schema
from trulens.core.utils.json import jsonify_for_ui
from trulens.core.utils.serial import JSON
from trulens.core.utils.serial import Lens
from trulens.dashboard.streamlit_utils import init_from_args
from trulens.dashboard.ux.apps import ChatRecord
from trulens.dashboard.ux.page_config import set_page_config

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

if __name__ == "__main__":
    # If not imported, gets args from command line and creates a TruSession
    init_from_args()

session = TruSession()

set_page_config(page_title="App Runner")


def remove_selector(
    container,
    type: str,  # either "app" or "record"
    selector_idx: int,
    record_idx: int,
    selector: str,
    rec: Optional[ChatRecord] = None,
):
    """
    Remove the `selector` of type `type`. A selector should be uniquely
    addressed/keyed by `type`, `selector_idx`, and `record_idx` but don't
    presently see a reason to have duplicate selectors so indexing only by
    `type` and `selector` for now. `container` is the streamlit "empty" object
    that contains the widgets for this selector. For `record` types,
    `record_idx` is ignored as the selector is removed from all of the
    rows/records.
    """

    state = st.session_state[f"selectors_{type}"]

    if selector in state:
        state.remove(selector)
    else:
        print("no such selector")
        return

    # Get and delete all of the containers for this selector. If this is a
    # record `type`, there will be one container for each record row.
    key_norec = f"{type}_{selector_idx}"
    for container in st.session_state[f"containers_{key_norec}"]:
        del container


def update_selector(
    container,
    type: str,
    selector_idx: int,
    record_idx: int,
    selector: str,
    rec: Optional[ChatRecord] = None,
):
    """
    Update the selector keyed by `type`, `selector_idx`, `record_idx` to the new
    value retrieved from state. Only works assuming selectors are unique within
    each `type`.
    """

    state = st.session_state[f"selectors_{type}"]

    key = f"{type}_{selector_idx}_{record_idx}"

    new_val = st.session_state[f"edit_input_{key}"]

    state[state.index(selector)] = new_val


def draw_selector(
    type: str,
    selector_idx: int,
    record_idx: int,
    selector: str,
    rec: Optional[ChatRecord] = None,
):
    """
    Draws the UI elements for a selector of type `type` intended to be keyed by
    (type) and `selector_idx` and `record_idx`. The selector represents a
    Lens given as str in `selector`. Includes delete and edit widgets as
    well as the listing of the values attained by the selected path in the given
    ChatRecord `rec`.
    """

    key = f"{type}_{selector_idx}_{record_idx}"
    key_norec = f"{type}_{selector_idx}"

    container = st.empty()

    # Add the container for these elements into the state indexed by type and
    # selector so we can easily delete it later alongside its analogues in order
    # records (for "record" `type` selectors).
    if f"containers_{key_norec}" not in st.session_state:
        st.session_state[f"containers_{key_norec}"] = []
    st.session_state[f"containers_{key_norec}"].append(container)

    # Cannot stack columns too deeply:
    # c1, c2 = st.columns(2)

    # TODO: figure out how to expand/collapse these across all records at the
    # same time, this session thing does not work.
    st.session_state[f"expanded_{key_norec}"] = True

    # Put everything in this expander:
    exp = container.expander(
        label=selector, expanded=st.session_state[f"expanded_{key_norec}"]
    )

    # Edit input.
    exp.text_input(
        label="App selector",
        value=selector,
        key=f"edit_input_{key}",
        on_change=update_selector,
        kwargs=dict(
            container=container,
            type=type,
            selector_idx=selector_idx,
            record_idx=record_idx,
            selector=selector,
            rec=rec,
        ),
        label_visibility="collapsed",
    )

    # Get the relevant JSON to path into.
    obj = rec.app_json
    if type == "record":
        obj = mod_record_schema.Record.model_validate(
            rec.record_json
        ).layout_calls_as_app()

    # Try to parse the selector as a Lens.
    path = None
    try:
        path = Lens.of_string(selector)
    except Exception as e:
        exp.write(f"parse error: {e}")

    if path is not None:
        try:
            # Draw each value addressed by `path`:
            for val in path.get(obj):
                json_val = jsonify_for_ui(val)
                if isinstance(json_val, dict):
                    # Don't expand by default as there are large JSONs and
                    # streamlit scrolls to the bottom if shown.
                    exp.json(json_val, expanded=False)
                else:
                    exp.write(json_val)

        except Exception as e:
            exp.write(f"enumeration error: {e}")


def draw_rec(record_idx: int, rec: ChatRecord, skip_human: bool = False):
    """
    Draw a ChatRecord `rec`, indexed by `record_idx` into columns `cols`. If
    `skip_human` is True, skips the human part of the record (as it was already
    drawn by an earlier call).
    """

    record_json = rec.record_json

    if not skip_human:
        if rec.human is not None:
            with st.chat_message("Human", avatar="🧑‍💻"):
                st.write(rec.human)

    if rec.computer is not None:
        with st.chat_message("Computer", avatar="🤖"):
            st.write(rec.computer)

    if record_json is not None:
        # st.write(f"TODO link to {record_json['record_id']}.")

        for selector_idx, selector in enumerate(
            st.session_state.selectors_record
        ):
            draw_selector(
                type="record",
                selector_idx=selector_idx,
                record_idx=record_idx,
                selector=selector,
                rec=rec,
            )


def set_selector(type: str):
    """
    Set the selectors of type `type` in session state.
    """

    # Get value from session:
    input_key = f"set_{type}_selector_input"
    val = st.session_state[input_key]

    st.session_state[f"selectors_{type}"] = val


def add_selector(type: str):
    """
    Add a new selector of type `type`. Value is looked up from session state.
    """

    # Get value from session:
    input_key = f"add_{type}_selector_input"
    val = st.session_state[input_key]

    # The list of selectors of the appropriate type:
    state = st.session_state[f"selectors_{type}"]

    # Add selector if not already in there:
    if val not in state:
        state.append(val)

        # Clear input
        st.session_state[input_key] = ""
    else:
        print(f"{type} selector {val} already exists")


def select_app(app_json: JSON):
    """
    Select the app to start a session with by its JSON.
    """

    tru_app = mod_app_schema.AppDefinition.new_session(
        app_definition_json=app_json
    )

    st.session_state.records = [ChatRecord(app_json=app_json, app=tru_app)]

    for typ in ["app", "record"]:
        st.session_state[f"selectors_{typ}"] = []


def run_record(col):
    """
    Assuming a user provided some input in the appropriate text box, run the app
    from its final state on the new input.
    """

    # Current is last:
    last_record_index = len(st.session_state.records) - 1
    current_record = st.session_state.records[last_record_index]

    # Get the human input from state and update record.
    human_input = st.session_state["human_input"]
    current_record.human = human_input

    # Draw the ChatRecord so far, just human input.
    with col:
        draw_rec(record_idx=last_record_index, rec=current_record)

    # TODO: set some sort of progress bar or do async computation for computer
    # response.
    tru_app = current_record.app

    # Run the app and collect the record.
    with tru_app as rec:
        comp_response = tru_app.main_call(human_input)
    record: mod_record_schema.Record = rec.get()

    # Update ChatRecord.
    current_record.computer = comp_response

    # Doing this after draw_rec so that the computer output can show up before
    # we start rendering selected values.
    current_record.record_json = record.model_dump()

    # Add the next ChatRecord that contains the updated app state:
    st.session_state.records.append(
        ChatRecord(app=tru_app, app_json=tru_app.model_dump())
    )


def end_session():
    """
    Reset the state to that before a session is started.
    """

    del st.session_state["records"]


if "records" not in st.session_state:
    # This field only exists after a model is selected. Here no model was
    # selected yet. Show all loadable models for which a new session can be
    # started.

    st.title("App Runner")

    loadable_apps = mod_app_schema.AppDefinition.get_loadable_apps()

    st.divider()

    for app_json in loadable_apps:
        st.subheader(app_json["app_id"])
        st.button(
            label="New Session",
            key=f"select_app_{app_json['app_id']}",
            on_click=select_app,
            args=(app_json,),
        )

        st.divider()

    if len(loadable_apps) == 0:
        st.write(
            "No loadable apps found in database. "
            "To make an app loadable, specify a loader function via the `initial_app_loader` argument when wrapping the app. "
            "See the notebook at https://github.com/truera/trulens/blob/main/examples/experimental/streamlit_appui_example.ipynb for an example."
        )

else:
    # Otherwise a model was selected, and there should be at least one
    # ChatRecord in the state.

    first_record = st.session_state.records[0]
    app_json = first_record.app_json

    # Show the app id and some app-level or session-level controls/links.
    st.title(f"App Runner: {app_json['app_id']}")

    st.button(label="End session", on_click=end_session)

    # st.write(f"TODO: link to {app_json['app_id']} on other pages.")

    st.divider()

    left, right = st.columns([1 / 3, 2 / 3])

    with left:
        # On the left are app selectors that show the properties of the app as
        # it is at the current/final state of the session.

        st.write("#### App details")
        st.caption(
            "Details about your app. Use app selectors below to select the pieces of information to highlight."
        )

        # Create an add app selector input:
        st.write("**App selectors**")
        if len(st.session_state.selectors_app) > 0:
            st.multiselect(
                "Current app selectors",
                st.session_state.selectors_app,
                st.session_state.selectors_app,
                on_change=set_selector,
                key="set_app_selector_input",
                args=("app",),
                label_visibility="collapsed",
            )
        st.text_input(
            label="add app selector",
            key="add_app_selector_input",
            placeholder="Add an app selector (e.g. app.llm.model_name)",
            on_change=add_selector,
            args=("app",),
            label_visibility="collapsed",
        )

        # Draw existing app selectors.
        for i, selector in enumerate(st.session_state.selectors_app):
            draw_selector(
                type="app",
                selector_idx=i,
                record_idx=None,
                selector=selector,
                rec=first_record,
            )

    with right:
        # On the right 2/3 are rows, one per ChatRecord.
        st.write("#### Records")
        st.caption(
            "Your interactive chat session. Type in a message below to 'chat' with the LLM application. Record selectors can be used to surface information on a per-record level."
        )

        st.write("**Record selectors**")
        if len(st.session_state.selectors_record) > 0:
            st.multiselect(
                "Current record selectors",
                st.session_state.selectors_record,
                st.session_state.selectors_record,
                on_change=set_selector,
                key="set_record_selector_input",
                args=("record",),
                label_visibility="collapsed",
            )
        st.text_input(
            label="add record selector",
            placeholder="Add a record selector to view details about the records (e.g. cost.cost).",
            key="add_record_selector_input",
            on_change=add_selector,
            args=("record",),
            label_visibility="collapsed",
        )

        # Rows corresponding to ChatRecord:
        for i, rec in enumerate(st.session_state.records):
            draw_rec(record_idx=i, rec=rec)

        if (
            len(st.session_state.records) == 0
            or st.session_state.records[0].record_json is None
        ):
            st.caption("Begin a chat session by typing in a message below")

    # NOTE: chat input cannot be inside column.
    human_input = st.chat_input(
        on_submit=run_record,
        key="human_input",
        kwargs=dict(
            col=right  # should be the cols of the last row from the above enumeration.
        ),
    )

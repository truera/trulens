import difflib
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from trulens.core.schema import prompt as prompt_schema
from trulens.dashboard.constants import PROMPTS_PAGE_NAME as page_name
from trulens.dashboard.utils.dashboard_utils import get_prompt_label_history
from trulens.dashboard.utils.dashboard_utils import get_prompt_labels
from trulens.dashboard.utils.dashboard_utils import get_prompt_version
from trulens.dashboard.utils.dashboard_utils import get_prompt_versions
from trulens.dashboard.utils.dashboard_utils import get_prompts
from trulens.dashboard.utils.dashboard_utils import get_session
from trulens.dashboard.utils.dashboard_utils import (
    read_query_params_into_session_state,
)
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.streamlit_compat import st_columns

DEFAULT_CHAT_TEMPLATE = (
    '[\n  {"role": "system", "content": ""},\n'
    '  {"role": "user", "content": "{{question}}"}\n]'
)


def init_page_state():
    if st.session_state.get(f"{page_name}.initialized", False):
        return

    read_query_params_into_session_state(page_name=page_name)

    st.session_state[f"{page_name}.initialized"] = True


def _version_label_map(labels_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map each version id to the labels currently pointing at it."""

    mapping: Dict[str, List[str]] = {}
    if labels_df is None or labels_df.empty:
        return mapping

    for _, row in labels_df.iterrows():
        mapping.setdefault(row["version_id"], []).append(row["label"])
    for labels in mapping.values():
        labels.sort()
    return mapping


def _version_options(
    versions_df: pd.DataFrame, labels_df: pd.DataFrame
) -> Dict[str, str]:
    """Map each version id to a one-line description for a select box."""

    label_map = _version_label_map(labels_df)

    options: Dict[str, str] = {}
    for _, row in versions_df.iterrows():
        version_id = row["version_id"]
        labels = label_map.get(version_id, [])
        suffix = f" [{', '.join(labels)}]" if labels else ""
        note = row.get("change_note")
        if note is None or pd.isna(note) or not str(note).strip():
            note = "no change note"
        options[version_id] = f"{version_id[:20]}… {note}{suffix}"
    return options


def _version_lines(version: prompt_schema.PromptVersion) -> List[str]:
    """Flatten a version into comparable lines of text."""

    if version.prompt_type is prompt_schema.PromptType.TEXT:
        return (version.text or "").splitlines() or [""]

    lines: List[str] = []
    for message in version.messages or []:
        lines.append(f"[{message.role.value}]")
        if isinstance(message.content, str):
            lines.extend(message.content.splitlines() or [""])
        else:
            lines.extend(
                json.dumps(
                    message.content, indent=2, sort_keys=True
                ).splitlines()
            )
    return lines


def _structured_diff(
    left: prompt_schema.PromptVersion,
    right: prompt_schema.PromptVersion,
) -> str:
    """Unified diff between two versions."""

    return "\n".join(
        difflib.unified_diff(
            _version_lines(left),
            _version_lines(right),
            fromfile=left.version_id,
            tofile=right.version_id,
            lineterm="",
        )
    )


def _preview(
    version: prompt_schema.PromptVersion,
    prompt: prompt_schema.Prompt,
    values: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Render a local preview, returning `(preview, error)`.

    No model is invoked and no credentials are read.
    """

    resolved = prompt_schema.ResolvedPrompt(prompt=prompt, version=version)
    try:
        rendered = resolved.render(**values)
    except prompt_schema.VariableError as e:
        return None, str(e)

    if rendered.text is not None:
        return rendered.text, None
    return json.dumps(rendered.messages, indent=2), None


def _render_version_editor(prompt: prompt_schema.Prompt):
    st.subheader("Create a version")
    st.caption(
        "Versions are immutable and content-addressed. Saving identical "
        "content returns the existing version."
    )

    change_note = st.text_input("Change note", key=f"{page_name}.change_note")

    if prompt.prompt_type is prompt_schema.PromptType.TEXT:
        text = st.text_area("Template", key=f"{page_name}.new_text", height=200)
        messages = None
    else:
        text = None
        raw = st.text_area(
            "Messages (JSON list of {role, content})",
            key=f"{page_name}.new_messages",
            height=240,
            value=DEFAULT_CHAT_TEMPLATE,
        )
        try:
            messages = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError as e:
            st.error(f"Messages must be valid JSON: {e}")
            return

    if st.button("Create version", key=f"{page_name}.create_version"):
        session = get_session()
        try:
            version = session.create_prompt_version(
                prompt=prompt,
                text=text or None,
                messages=messages,
                change_note=change_note or None,
            )
        except (ValueError, prompt_schema.InvalidTemplateError) as e:
            st.error(str(e))
            return
        st.cache_data.clear()
        st.success(f"Created version {version.version_id}")
        st.rerun()


def _render_label_controls(
    prompt: prompt_schema.Prompt,
    versions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
):
    st.subheader("Labels")
    st.caption(
        "A label is a mutable pointer to one exact version. Rolling back is "
        "moving a label to an older version; versions are never edited."
    )

    if labels_df is not None and not labels_df.empty:
        st.dataframe(labels_df, hide_index=True, width="stretch")

    options = _version_options(versions_df, labels_df)

    col_label, col_version = st_columns([1, 2])
    with col_label:
        label = st.text_input(
            "Label", value="production", key=f"{page_name}.move_label"
        )
    with col_version:
        target = st.selectbox(
            "Version",
            options=list(options),
            format_func=lambda vid: options[vid],
            key=f"{page_name}.move_version",
        )

    if st.button("Move label", key=f"{page_name}.move_label_button"):
        session = get_session()
        try:
            session.set_prompt_label(
                prompt, label=label, version=target, moved_by="dashboard"
            )
        except ValueError as e:
            st.error(str(e))
            return
        st.cache_data.clear()
        st.success(f"Label {label!r} now points at {target}")
        st.rerun()


def _render_diff(versions_df: pd.DataFrame, labels_df: pd.DataFrame):
    st.subheader("Diff")

    if len(versions_df) < 2:
        st.info("Create a second version to see a diff.")
        return

    label_map = _version_label_map(labels_df)
    options = _version_options(versions_df, labels_df)
    ids = list(options)

    col_left, col_right = st_columns(2)
    with col_left:
        left_id = st.selectbox(
            "Baseline",
            options=ids,
            index=len(ids) - 2,
            format_func=lambda vid: options[vid],
            key=f"{page_name}.diff_left",
        )
    with col_right:
        right_id = st.selectbox(
            "Candidate",
            options=ids,
            index=len(ids) - 1,
            format_func=lambda vid: options[vid],
            key=f"{page_name}.diff_right",
        )

    left = get_prompt_version(left_id)
    right = get_prompt_version(right_id)
    if left is None or right is None:
        st.error("One of the selected versions is missing.")
        return

    for version_id in (left_id, right_id):
        labels = label_map.get(version_id)
        if labels:
            st.caption(f"{version_id[:20]}… carries {', '.join(labels)}")

    diff = _structured_diff(left, right)
    if not diff:
        st.info("The two versions have identical content.")
    else:
        st.code(diff, language="diff")


def _render_preview(prompt: prompt_schema.Prompt, versions_df: pd.DataFrame):
    st.subheader("Preview")
    st.caption(
        "Rendering happens in this process. No model is called and no "
        "credentials are read or stored."
    )

    version_id = st.selectbox(
        "Version",
        options=list(versions_df["version_id"]),
        index=len(versions_df) - 1,
        key=f"{page_name}.preview_version",
    )
    version = get_prompt_version(version_id)
    if version is None:
        st.error("Version not found.")
        return

    values = {}
    for name in version.variables:
        values[name] = st.text_input(name, key=f"{page_name}.preview.{name}")

    preview, error = _preview(version, prompt, values)
    if error:
        st.warning(error)
    else:
        st.code(preview)


def render_prompts():
    prompts_df = get_prompts()

    if prompts_df is None or prompts_df.empty:
        st.info(
            "No prompts found. Create one with "
            "`session.create_prompt(slug=..., prompt_type=...)`."
        )
        return

    st.dataframe(prompts_df, hide_index=True, width="stretch")

    slug = st.selectbox(
        "Prompt",
        options=list(prompts_df["slug"]),
        key=f"{page_name}.slug",
    )

    session = get_session()
    prompt = session.connector.db.get_prompt(slug=slug)
    if prompt is None:
        st.error(f"Prompt {slug!r} not found.")
        return

    versions_df = get_prompt_versions(prompt.prompt_id)
    labels_df = get_prompt_labels(prompt.prompt_id)

    st.subheader("Versions")
    if versions_df is None or versions_df.empty:
        st.info("This prompt has no versions yet.")
        _render_version_editor(prompt)
        return

    st.dataframe(versions_df, hide_index=True, width="stretch")

    _render_label_controls(prompt, versions_df, labels_df)
    _render_diff(versions_df, labels_df)
    _render_preview(prompt, versions_df)
    _render_version_editor(prompt)

    st.subheader("Label history")
    st.dataframe(
        get_prompt_label_history(prompt.prompt_id),
        hide_index=True,
        width="stretch",
    )


def prompts_main():
    set_page_config(page_title=page_name)
    init_page_state()
    render_prompts()


if __name__ == "__main__":
    prompts_main()

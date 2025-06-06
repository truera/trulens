from typing import Optional

from packaging.version import Version
import streamlit as st

st_version = Version(st.__version__)

st_dialog = st.dialog if hasattr(st, "dialog") else st.experimental_dialog
st_fragment = (
    st.fragment if hasattr(st, "fragment") else st.experimental_fragment
)


def st_columns(
    spec, *, gap: str = "small", vertical_alignment: str = "top", container=None
):
    container = container or st
    if st_version >= Version("1.36.0"):
        return container.columns(
            spec, gap=gap, vertical_alignment=vertical_alignment
        )
    else:
        return container.columns(spec, gap=gap)


def st_code(
    body,
    language: Optional[str] = None,
    *,
    line_numbers: bool = False,
    wrap_lines: bool = False,
    container=None,
):
    container = container or st
    if st_version >= Version("1.38.0"):
        return container.code(
            body,
            language=language,
            line_numbers=line_numbers,
            wrap_lines=wrap_lines,
        )
    else:
        return container.code(
            body,
            language=language,
            line_numbers=line_numbers,
        )

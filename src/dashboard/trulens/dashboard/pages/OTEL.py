import streamlit as st
from trulens.dashboard.components.record_viewer_otel import record_viewer_otel
from trulens.dashboard.utils.dashboard_utils import get_spans
from trulens.dashboard.utils.dashboard_utils import set_page_config


def otel_main():
    set_page_config("OTEL")
    st.header("test")
    spans_df = get_spans("test")
    spans = spans_df.to_dict(orient="records")
    record_viewer_otel(spans)


if __name__ == "__main__":
    otel_main()

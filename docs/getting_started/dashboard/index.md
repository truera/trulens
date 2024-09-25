# Viewing Results

TruLens provides a broad set of capabilities for evaluating and tracking applications. In addition, TruLens ships with native tools for examining traces and evaluations in the form of a complete dashboard, and components that can be added to streamlit apps.

## TruLens Dashboard

To view and examine application logs and feedback results, TruLens provides a built-in Streamlit dashboard. That app has two pages, the Leaderboard which displays aggregate feedback results and metadata for each application version, and the Evaluations page where you can more closely examine individual traces and feedback results. This dashboard is launched by [run_dashboard][trulens.dashboard.run_dashboard], and will run from a database url you specify with  [TruSession()][trulens.core.TruSession].

!!! example "Launch the TruLens dashboard"

    ```python
    from trulens.dashboard import run_dashboard
    session = TruSession(database_url = ...) # or default.sqlite by default
    run_dashboard(session)
    ```

By default, the dashboard will find and run on an unused port number. You can also specify a port number for the dashboard to run on. The function will output a link where the dashboard is running.

!!! example "Specify a port"

    ```python
    from trulens.dashboard import run_dashboard
    run_dashboard(port=8502)
    ```

!!! note

    If you are running in Google Colab, `run_dashboard()` will output a tunnel website and IP address that can be entered into the tunnel website.


## Streamlit Components

In addition to the complete dashboard, several of the dashboard components can be used on their own and added to existing _Streamlit_ dashboards.

_Streamlit_ is an easy way to create python scripts into shareable web applications, and has become a popular way to interact with generative AI technology. Several _TruLens_ UI components are now accessible for adding to Streamlit dashboards using the _TruLens_ [Streamlit module][trulens.dashboard.streamlit].

Consider the below `app.py` which consists of a simple RAG application that is already logged and evaluated with _TruLens_. Notice in particular, that we are getting both the application's `response` and `record`.

!!! example "Simple Streamlit app with TruLens"

    ```python

    import streamlit as st
    from trulens.core import TruSession

    from base import rag # a rag app with a query method
    from base import tru_rag # a rag app wrapped by trulens

    session = TruSession()

    def generate_and_log_response(input_text):
        with tru_rag as recording:
            response = rag.query(input_text)
        record = recording.get()
        return record, response

    with st.form("my_form"):
        text = st.text_area("Enter text:", "How do I launch a streamlit app?")
        submitted = st.form_submit_button("Submit")
        if submitted:
            record, response = generate_and_log_response(text)
            st.info(response)

    ```

With the `record` in hand, we can easily add TruLens components to display the evaluation results of the provided record using [trulens_feedback][trulens.dashboard.streamlit.trulens_feedback]. This will display the _TruLens_ feedback result clickable pills as the feedback is available.

!!! example "Display feedback results"

    ```python
    from trulens.dashboard import streamlit as trulens_st

    if submitted:
        trulens_st.trulens_feedback(record=record)
    ```

In addition to the feedback results, we can also display the record's trace to help with debugging using [trulens_trace][trulens.dashboard.streamlit.trulens_trace] from the _TruLens_ streamlit module.

!!! example "Display the trace"

    ```python
    from trulens.dashboard import streamlit as trulens_st

    if submitted:
        trulens_st.trulens_trace(record=record)
    ```

Last, we can also display the TruLens leaderboard using [render_leaderboard][trulens.dashboard.Leaderboard.render_leaderboard] from the _TruLens_ streamlit module to understand the aggregate performance across application versions.

!!! example "Display the application leaderboard"

    ```python
    from trulens.dashboard.Leaderboard import render_leaderboard

    render_leaderboard()
    ```

In combination, the streamlit components allow you to make evaluation front-and-center in your app. This is particularly useful for developer playground use cases, or to ensure users of app reliability.

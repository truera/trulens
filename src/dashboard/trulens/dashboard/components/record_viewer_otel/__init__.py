import os
from typing import Any, Dict, List, Optional, Union

import streamlit.components.v1 as components
from typing_extensions import TypedDict

# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
# (This is, of course, optional - there are innumerable ways to manage your
# release process.)
_RELEASE = True

# Declare a Streamlit component. `declare_component` returns a function
# that is used to create instances of the component. We're naming this
# function "_component_func", with an underscore prefix, because we don't want
# to expose it directly to users. Instead, we will create a custom wrapper
# function, below, that will serve as our component's public API.

# It's worth noting that this call to `declare_component` is the
# *only thing* you need to do to create the binding between Streamlit and
# your component frontend. Everything else we do in this file is simply a
# best practice.
_record_viewer_otel = None


class SpanTrace(TypedDict):
    trace_id: str
    parent_id: str
    span_id: str


class SpanRecord(TypedDict):
    name: str
    parent_span_id: str
    status: str


class OtelSpan(TypedDict):
    event_id: str
    record: SpanRecord
    record_attributes: Dict[str, Any]
    start_timestamp: Union[int, float]
    timestamp: Union[int, float]
    trace: SpanTrace


# Create a wrapper function for the component. This is an optional
# best practice - we could simply expose the component function returned by
# `declare_component` and call it done. The wrapper allows us to customize
# our component's API: we can pre-process its input args, post-process its
# output value, and add a docstring for users.
def record_viewer_otel(
    spans: List[OtelSpan], key: Optional[str] = None
) -> None:
    """Create a new instance of "record_viewer_otel", which produces a record viewer for the OTEL spans.

    Args:
        spans: List of spans to be displayed in the timeline. It is the caller's responsibility
               to select the spans to be displayed. The simplest way to get the spans is to
               get the rows from the ORM.Events table, then call to_dict(orient="records") on the rows.
    """

    # Call through to our private component function. Arguments we pass here
    # will be sent to the frontend, where they'll be available in an "args"
    # dictionary.
    #
    # "default" is a special argument that specifies the initial return
    # value of the component before the user has interacted with it.
    global _record_viewer_otel

    if _record_viewer_otel is None:
        if not _RELEASE:
            _record_viewer_otel = components.declare_component(
                # We give the component a simple, descriptive name
                "record_viewer_otel",
                # Pass `url` here to tell Streamlit that the component will be served
                # by the local dev server that you run via `npm run start`.
                # (This is useful while your component is in development.)
                url="http://localhost:5173",
            )
        else:
            # When we're distributing a production version of the component, we'll
            # replace the `url` param with `path`, and point it to to the component's
            # build directory:
            parent_dir = os.path.dirname(os.path.abspath(__file__))
            build_dir = os.path.join(parent_dir, "dist")
            _record_viewer_otel = components.declare_component(
                "record_viewer_otel", path=build_dir
            )

    _record_viewer_otel(spans=spans, key=key, default="")

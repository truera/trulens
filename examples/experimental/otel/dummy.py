# ruff: noqa: F401

from pathlib import Path
from pprint import pprint
import sys

import dotenv
from opentelemetry import trace

# zipkip exporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter

# console exporter
from trulens.apps.custom import TruCustomApp
from trulens.core import Feedback
from trulens.core import Select
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.trace import TracerProvider
from trulens.feedback.dummy.provider import DummyProvider

# Add base dir to path to be able to access test folder.
base_dir = Path().cwd().parent.parent.parent.resolve()
if str(base_dir) not in sys.path:
    print(f"Adding {base_dir} to sys.path")
    sys.path.append(str(base_dir))


from examples.dev.dummy_app.app import DummyApp  # noqa: E402

dotenv.load_dotenv()

# Sets the global default tracer provider to be the trulens one.
trace.set_tracer_provider(TracerProvider())

# Creates a tracer for custom spans below.
tracer = trace.get_tracer(__name__)

# Setup zipkin exporter
exporter = ZipkinExporter(endpoint="http://localhost:9411/api/v2/spans")

# Setup session with exporter.
session = TruSession(_experimental_otel_exporter=exporter)

# If not using the exporter, manually enable the otel experimental feature:
# session = TruSession()
# session.experimental_enable_feature("otel_tracing")

# Create dummy endpoint for a dummy feedback function:
dummy_provider = DummyProvider()
dummy_feedback = Feedback(dummy_provider.sentiment).on(
    text=Select.RecordSpans.trulens.call.generate.attributes[
        "trulens.bindings"
    ].prompt
)
# Parts of the selector are:
#
# - Select.RecordSpans - Select spans dictionary, organized by span name.
#
# - trulens.call.generate - Span name. TruLens call spans are of the form
#   "trulens.call.<method_name>".
#
# - attributes - Span attributes
#
# - ["trulens.bindings"] - Attributes specific to TruLens spans. Call spans include
#   method call arguments in "trulens.bindings".
#
# - prompt - The prompt argument to the method call named.

# Create custom app:
ca = DummyApp()

# Create trulens wrapper:
ta = TruCustomApp(
    ca,
    app_id="customapp",
    feedbacks=[dummy_feedback],
)

# Normal trulens recording context manager:
with ta as recorder:
    # Another custom span.
    with tracer.start_as_current_span("custom inner span") as inner_span:
        inner_span.set_attribute("custom", "value")

        # Normal instrumented call:
        print(ca.respond_to_query("hello"))

record = recorder.get()

# Check the feedback results. Note that this feedback function is from a dummy
# provider which does not have true sentiment analysis.

pprint(record.feedback_results[0].result())

# Check trulens instrumented calls as spans:

pprint(record.get(Select.RecordSpans))

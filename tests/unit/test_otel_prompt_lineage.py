"""Prompt lineage on generation spans (issue #2703)."""

import os
import unittest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from trulens.core.otel.instrument import prompt_lineage
from trulens.core.otel.instrument import set_prompt_lineage_attributes
from trulens.core.schema import prompt as prompt_schema
from trulens.experimental.otel_tracing.core.span import (
    is_content_capture_enabled,
)
from trulens.experimental.otel_tracing.core.span import set_content_capture
from trulens.otel.semconv.trace import SpanAttributes

SECRET_QUESTION = "my card number is 4111111111111111"


def _rendered(label=None):
    prompt = prompt_schema.Prompt(slug="support-assistant", prompt_type="chat")
    version = prompt_schema.PromptVersion(
        prompt_id=prompt.prompt_id,
        prompt_type="chat",
        messages=[
            {"role": "system", "content": "Answer using the support policy."},
            {"role": "user", "content": "{{question}}"},
        ],
        variables=["question"],
    )
    resolved = prompt_schema.ResolvedPrompt(
        prompt=prompt, version=version, label=label
    )
    return prompt, version, resolved.render(question=SECRET_QUESTION)


class TestPromptLineage(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = provider.get_tracer(__name__)
        self.addCleanup(set_content_capture, None)

    def _attributes(self, rendered):
        with self.tracer.start_as_current_span("generation"):
            with prompt_lineage(rendered) as handed_back:
                self.assertIs(handed_back, rendered)
        (span,) = self.exporter.get_finished_spans()
        return dict(span.attributes)

    def test_lineage_lands_on_the_current_span(self):
        prompt, version, rendered = _rendered(label="production")

        attributes = self._attributes(rendered)

        self.assertEqual(attributes[SpanAttributes.PROMPT.ID], prompt.prompt_id)
        self.assertEqual(
            attributes[SpanAttributes.PROMPT.SLUG], "support-assistant"
        )
        self.assertEqual(
            attributes[SpanAttributes.PROMPT.VERSION_ID], version.version_id
        )
        self.assertEqual(attributes[SpanAttributes.PROMPT.LABEL], "production")
        self.assertEqual(
            attributes[SpanAttributes.PROMPT.RENDERED_CONTENT_HASH],
            rendered.rendered_content_hash,
        )

    def test_label_is_absent_for_an_exact_version_request(self):
        _, _, rendered = _rendered(label=None)

        attributes = self._attributes(rendered)

        self.assertNotIn(SpanAttributes.PROMPT.LABEL, attributes)

    def test_lineage_works_with_content_capture_disabled(self):
        set_content_capture(False)
        _, version, rendered = _rendered(label="production")

        attributes = self._attributes(rendered)

        self.assertFalse(is_content_capture_enabled())
        self.assertEqual(
            attributes[SpanAttributes.PROMPT.VERSION_ID], version.version_id
        )

    def test_lineage_works_with_content_capture_enabled(self):
        set_content_capture(True)
        _, version, rendered = _rendered(label="production")

        attributes = self._attributes(rendered)

        self.assertTrue(is_content_capture_enabled())
        self.assertEqual(
            attributes[SpanAttributes.PROMPT.VERSION_ID], version.version_id
        )

    def test_prompt_body_is_never_copied_into_attributes(self):
        for enabled in (False, True):
            with self.subTest(content_capture=enabled):
                set_content_capture(enabled)
                self.exporter.clear()
                _, _, rendered = _rendered(label="production")

                attributes = self._attributes(rendered)

                lineage = {
                    key: value
                    for key, value in attributes.items()
                    if key.startswith(SpanAttributes.PROMPT.base)
                }
                self.assertTrue(lineage)
                for value in lineage.values():
                    self.assertNotIn(SECRET_QUESTION, str(value))
                    self.assertNotIn("support policy", str(value))

    def test_env_var_gate_is_untouched_by_lineage(self):
        set_content_capture(None)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_content_capture_enabled())
            _, version, rendered = _rendered()

            attributes = self._attributes(rendered)

            self.assertEqual(
                attributes[SpanAttributes.PROMPT.VERSION_ID],
                version.version_id,
            )

    def test_non_recording_span_is_ignored(self):
        _, _, rendered = _rendered()

        class _NotRecording:
            def is_recording(self):
                return False

            def set_attribute(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("should not write to a stopped span")

        set_prompt_lineage_attributes(_NotRecording(), rendered)
        set_prompt_lineage_attributes(None, rendered)


if __name__ == "__main__":
    unittest.main()

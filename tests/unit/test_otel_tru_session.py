import os
from typing import List
from unittest import mock

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession

from tests.util.otel_test_case import OtelTestCase

TRULENS_SERVICE_NAME = "trulens"


class TestOtelTruSession(OtelTestCase):
    def test_resource_attributes(self):
        """Tracer provider resource should carry service.name,
        service.version, and — when configured — deployment.environment."""
        provider = trace.get_tracer_provider()
        resource = provider.resource
        self.assertEqual(resource.attributes["service.name"], "trulens")
        self.assertIn("service.version", resource.attributes)
        self.assertIsInstance(resource.attributes["service.version"], str)
        self.assertTrue(len(resource.attributes["service.version"]) > 0)

    def test_deployment_environment_absent_by_default(self):
        """deployment.environment should be absent from the resource
        when no TruLens-specific env var is set."""
        provider = trace.get_tracer_provider()
        self.assertNotIn("deployment.environment", provider.resource.attributes)

    def test_deployment_environment_otel_resource_attributes_passthrough(self):
        """OTEL_RESOURCE_ATTRIBUTES=deployment.environment=production
        should be picked up by Resource.create().

        We test Resource.create() directly because
        trace.set_tracer_provider() only takes effect once per process,
        so calling _set_up_tracer_provider() a second time cannot
        replace the global provider.
        """
        with mock.patch.dict(
            os.environ,
            {
                "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment=production",
            },
            clear=False,
        ):
            resource = Resource.create({"service.name": TRULENS_SERVICE_NAME})
        self.assertEqual(
            resource.attributes["deployment.environment"],
            "production",
        )

    def test_get_records_and_feedback(self):
        # Create app.
        class _TestApp:
            @instrument()
            def query(self, question: str) -> str:
                return "Kojikun"

        app = _TestApp()
        tru_app = TruApp(
            app, app_name="test_get_records_and_feedback", app_version="v1"
        )
        # Record two separate records using two recording contexts.
        record_ids: List[str] = []
        with tru_app as recording1:
            app.query("Who is the best baby?")
            record_ids.extend([rec.record_id for rec in recording1.records])
        with tru_app as recording2:
            app.query("Who is the cutest baby?")
            record_ids.extend([rec.record_id for rec in recording2.records])
        # Get records and feedback.
        tru_session = TruSession()
        tru_session.force_flush()

        def num_records(record_ids: List[str]) -> int:
            return len(
                tru_session.get_records_and_feedback(record_ids=record_ids)[0]
            )

        self.assertEqual(2, len(record_ids))
        self.assertEqual(2, num_records(record_ids))
        self.assertEqual(1, num_records([record_ids[0]]))
        self.assertEqual(1, num_records([record_ids[1]]))
        self.assertEqual(0, num_records([]))

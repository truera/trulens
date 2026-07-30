from trulens.apps.app import TruApp
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class TestOtelNestedRecordRoot(OtelTestCase):
    def test_nested_tru_app_root_links_to_outer_record_root(self):
        class InnerApp:
            def query(self, question: str) -> str:
                return f"inner: {question}"

        inner_app = InnerApp()
        inner_tru_app = TruApp(
            app=inner_app,
            app_name="inner_app",
            app_version="v1",
            main_method=inner_app.query,
        )

        class OuterApp:
            def __init__(self, inner_tru_app):
                self.inner_tru_app = inner_tru_app

            def query(self, question: str) -> str:
                return self.inner_tru_app.instrumented_invoke_main_method(
                    run_name="inner run",
                    input_id="inner input",
                    main_method_args=(question,),
                )

        outer_app = OuterApp(inner_tru_app)
        outer_tru_app = TruApp(
            app=outer_app,
            app_name="outer_app",
            app_version="v1",
            main_method=outer_app.query,
        )

        outer_tru_app.instrumented_invoke_main_method(
            run_name="outer run",
            input_id="outer input",
            main_method_args=("hello",),
        )

        TruSession().force_flush()
        events = self._get_events()

        root_events = events[
            events["record_attributes"].apply(
                lambda attrs: attrs.get(SpanAttributes.SPAN_TYPE)
                in {
                    SpanAttributes.SpanType.RECORD_ROOT,
                    SpanAttributes.SpanType.NESTED_RECORD_ROOT,
                }
            )
        ]

        self.assertEqual(
            2,
            len(root_events),
            f"Expected one outer RECORD_ROOT and one inner NESTED_RECORD_ROOT, got:\n"
            f"{events[['record', 'record_attributes', 'resource_attributes', 'trace']]}",
        )

        outer_roots = root_events[
            root_events["resource_attributes"].apply(
                lambda attrs: attrs.get(ResourceAttributes.APP_NAME)
                == "outer_app"
            )
        ]
        inner_roots = root_events[
            root_events["resource_attributes"].apply(
                lambda attrs: attrs.get(ResourceAttributes.APP_NAME)
                == "inner_app"
            )
        ]

        self.assertEqual(1, len(outer_roots))
        self.assertEqual(1, len(inner_roots))

        outer_root = outer_roots.iloc[0]
        inner_root = inner_roots.iloc[0]

        self.assertEqual(
            SpanAttributes.SpanType.RECORD_ROOT,
            outer_root["record_attributes"][SpanAttributes.SPAN_TYPE],
        )
        self.assertEqual(
            SpanAttributes.SpanType.NESTED_RECORD_ROOT,
            inner_root["record_attributes"][SpanAttributes.SPAN_TYPE],
        )

        outer_record_id = outer_root["record_attributes"][
            SpanAttributes.RECORD_ID
        ]
        inner_record_id = inner_root["record_attributes"][
            SpanAttributes.RECORD_ID
        ]
        self.assertNotEqual(outer_record_id, inner_record_id)

        outer_app_id = outer_root["resource_attributes"][
            ResourceAttributes.APP_ID
        ]
        inner_app_id = inner_root["resource_attributes"][
            ResourceAttributes.APP_ID
        ]
        self.assertNotEqual(outer_app_id, inner_app_id)

        self.assertEqual(
            outer_root["trace"]["span_id"],
            inner_root["record_attributes"][
                SpanAttributes.NESTED_RECORD_ROOT.PARENT_SPAN_ID
            ],
        )
        self.assertEqual(
            outer_app_id,
            inner_root["record_attributes"][
                SpanAttributes.NESTED_RECORD_ROOT.PARENT_APP_ID
            ],
        )

    def test_single_tru_app_without_outer_context_stays_record_root(self):
        class InnerApp:
            def query(self, question: str) -> str:
                return f"inner: {question}"

        inner_app = InnerApp()
        inner_tru_app = TruApp(
            app=inner_app,
            app_name="inner_app",
            app_version="v1",
            main_method=inner_app.query,
        )

        inner_tru_app.instrumented_invoke_main_method(
            run_name="inner run",
            input_id="inner input",
            main_method_args=("hello",),
        )

        TruSession().force_flush()
        events = self._get_events()

        root_events = events[
            events["record_attributes"].apply(
                lambda attrs: attrs.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.RECORD_ROOT
            )
        ]

        self.assertEqual(1, len(root_events))
        self.assertEqual(
            SpanAttributes.SpanType.RECORD_ROOT,
            root_events.iloc[0]["record_attributes"][SpanAttributes.SPAN_TYPE],
        )

    def test_unresolvable_parent_span_falls_back_to_record_root(self):
        from unittest.mock import Mock
        import uuid

        from opentelemetry.baggage import set_baggage
        import opentelemetry.context as context_api

        class InnerApp:
            def query(self, question: str) -> str:
                return f"inner: {question}"

        inner_app = InnerApp()
        inner_tru_app = TruApp(
            app=inner_app,
            app_name="inner_app",
            app_version="v1",
            main_method=inner_app.query,
        )

        parent_record_id = str(uuid.uuid4())
        parent_app_id = "outer_app_id"

        mock_recording = Mock()
        mock_recording.add_record_id = Mock()

        tokens = [
            context_api.attach(
                set_baggage(SpanAttributes.RECORD_ID, parent_record_id)
            ),
            context_api.attach(
                set_baggage(ResourceAttributes.APP_ID, parent_app_id)
            ),
            context_api.attach(set_baggage("__trulens_otel_ctx__", object())),
            context_api.attach(
                set_baggage("__trulens_recording__", mock_recording)
            ),
        ]

        try:
            inner_tru_app.instrumented_invoke_main_method(
                run_name="inner run",
                input_id="inner input",
                main_method_args=("hello",),
            )
        finally:
            while tokens:
                context_api.detach(tokens.pop())

        TruSession().force_flush()
        events = self._get_events()

        nested_roots = events[
            events["record_attributes"].apply(
                lambda attrs: attrs.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.NESTED_RECORD_ROOT
            )
        ]
        self.assertEqual(0, len(nested_roots))

        record_roots = events[
            events["record_attributes"].apply(
                lambda attrs: attrs.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.RECORD_ROOT
            )
        ]
        self.assertEqual(1, len(record_roots))

        fallback_root = record_roots.iloc[0]
        self.assertEqual(
            "inner_app",
            fallback_root["resource_attributes"][ResourceAttributes.APP_NAME],
        )
        self.assertNotEqual(
            parent_record_id,
            fallback_root["record_attributes"][SpanAttributes.RECORD_ID],
        )

    def test_baggage_only_parent_linkage_is_not_authoritative(self):
        import uuid

        from opentelemetry.baggage import set_baggage
        import opentelemetry.context as context_api

        class InnerApp:
            def query(self, question: str) -> str:
                return f"inner: {question}"

        inner_app = InnerApp()
        inner_tru_app = TruApp(
            app=inner_app,
            app_name="inner_app",
            app_version="v1",
            main_method=inner_app.query,
        )

        parent_record_id = str(uuid.uuid4())
        parent_app_id = "outer_app_id"

        tokens = [
            context_api.attach(
                set_baggage(SpanAttributes.RECORD_ID, parent_record_id)
            ),
            context_api.attach(
                set_baggage(ResourceAttributes.APP_ID, parent_app_id)
            ),
        ]

        try:
            inner_tru_app.instrumented_invoke_main_method(
                run_name="inner run",
                input_id="inner input",
                main_method_args=("hello",),
            )
        finally:
            while tokens:
                context_api.detach(tokens.pop())

        TruSession().force_flush()
        events = self._get_events()

        nested_roots = events[
            events["record_attributes"].apply(
                lambda attrs: attrs.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.NESTED_RECORD_ROOT
            )
        ]
        self.assertEqual(0, len(nested_roots))

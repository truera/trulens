"""Unit tests for generic LLM auto-instrumentation (#2627)."""

from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

from trulens.core import TruSession
from trulens.experimental.otel_tracing.core.auto_instrument import (
    auto_instrument_all_llms,
)
from trulens.experimental.otel_tracing.core.auto_instrument import (
    instrument_anthropic,
)
from trulens.experimental.otel_tracing.core.auto_instrument import (
    instrument_bedrock,
)
from trulens.experimental.otel_tracing.core.auto_instrument import (
    instrument_google,
)
from trulens.experimental.otel_tracing.core.auto_instrument import (
    instrument_litellm,
)
from trulens.experimental.otel_tracing.core.auto_instrument import (
    instrument_openai,
)
from trulens.otel.semconv.trace import SpanAttributes


class TestAutoInstrumentLLM(TestCase):
    def test_session_auto_instrument_llm_method_exists(self) -> None:
        """Verify TruSession exposes auto_instrument_llm method."""
        session = TruSession()
        self.assertTrue(hasattr(session, "auto_instrument_llm"))
        self.assertTrue(callable(session.auto_instrument_llm))

    def test_auto_instrument_idempotent(self) -> None:
        """Verify auto_instrument_all_llms is safe to call multiple times."""
        from trulens.experimental.otel_tracing.core.auto_instrument import (
            is_auto_instrumentation_enabled,
        )

        auto_instrument_all_llms()
        self.assertTrue(is_auto_instrumentation_enabled())
        auto_instrument_all_llms()
        self.assertTrue(is_auto_instrumentation_enabled())

    @patch(
        "trulens.experimental.otel_tracing.core.auto_instrument._can_import",
        return_value=True,
    )
    @patch("trulens.core.otel.instrument.instrument_method")
    def test_instrument_openai(
        self, mock_instrument_method: MagicMock, mock_can_import: MagicMock
    ) -> None:
        """Verify OpenAI auto-instrumentation hooks completions.create with GENERATION span."""
        instrument_openai()
        if mock_instrument_method.called:
            for call in mock_instrument_method.call_args_list:
                self.assertEqual(
                    call.kwargs.get("span_type"),
                    SpanAttributes.SpanType.GENERATION,
                )

    @patch("trulens.core.otel.instrument.instrument_method")
    def test_instrument_anthropic(
        self, mock_instrument_method: MagicMock
    ) -> None:
        """Verify Anthropic auto-instrumentation hooks messages.create with GENERATION span."""
        mock_messages = MagicMock()
        mock_messages.Messages = type("Messages", (), {"create": lambda: None})
        mock_messages.AsyncMessages = type(
            "AsyncMessages", (), {"create": lambda: None}
        )

        with (
            patch(
                "trulens.experimental.otel_tracing.core.auto_instrument._can_import",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "anthropic": MagicMock(),
                    "anthropic.resources": mock_messages,
                },
            ),
        ):
            instrument_anthropic()
            mock_instrument_method.assert_called()
            for call in mock_instrument_method.call_args_list:
                self.assertEqual(
                    call.kwargs.get("span_type"),
                    SpanAttributes.SpanType.GENERATION,
                )

    @patch(
        "trulens.experimental.otel_tracing.core.auto_instrument._can_import",
        return_value=True,
    )
    @patch("trulens.core.otel.instrument.instrument_method")
    def test_instrument_google(
        self, mock_instrument_method: MagicMock, mock_can_import: MagicMock
    ) -> None:
        """Verify Google GenAI auto-instrumentation hooks generate_content with GENERATION span."""
        instrument_google()
        if mock_instrument_method.called:
            for call in mock_instrument_method.call_args_list:
                self.assertEqual(
                    call.kwargs.get("span_type"),
                    SpanAttributes.SpanType.GENERATION,
                )

    @patch("trulens.core.otel.instrument.instrument_method")
    def test_instrument_bedrock(
        self, mock_instrument_method: MagicMock
    ) -> None:
        """Verify Bedrock auto-instrumentation hooks _make_api_call with GENERATION span."""
        mock_botocore = MagicMock()
        mock_botocore.client.BaseClient = type(
            "BaseClient", (), {"_make_api_call": lambda: None}
        )

        with (
            patch(
                "trulens.experimental.otel_tracing.core.auto_instrument._can_import",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "botocore": mock_botocore,
                    "botocore.client": mock_botocore.client,
                },
            ),
        ):
            instrument_bedrock()
            mock_instrument_method.assert_called()
            for call in mock_instrument_method.call_args_list:
                self.assertEqual(
                    call.kwargs.get("span_type"),
                    SpanAttributes.SpanType.GENERATION,
                )

    @patch(
        "trulens.experimental.otel_tracing.core.auto_instrument._can_import",
        return_value=True,
    )
    @patch("trulens.core.otel.instrument.instrument_method")
    def test_instrument_litellm(
        self, mock_instrument_method: MagicMock, mock_can_import: MagicMock
    ) -> None:
        """Verify LiteLLM auto-instrumentation hooks completion with GENERATION span."""
        instrument_litellm()
        if mock_instrument_method.called:
            for call in mock_instrument_method.call_args_list:
                self.assertEqual(
                    call.kwargs.get("span_type"),
                    SpanAttributes.SpanType.GENERATION,
                )

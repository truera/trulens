"""OTLP protocol selection for the client hooks `otlp` destination."""

from __future__ import annotations

import pytest
from trulens.core.otel.client_hooks import exporting


@pytest.fixture
def captured_session_kwargs(monkeypatch):
    """Capture the TruSession arguments instead of building exporters."""
    captured = {}

    def fake_session(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(exporting.core_session, "TruSession", fake_session)
    monkeypatch.setenv("TRULENS_DESTINATION", "otlp")
    for name in (
        "TRULENS_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    ):
        monkeypatch.delenv(name, raising=False)
    return captured


def test_standard_protocol_variables_are_left_to_the_session(
    captured_session_kwargs, monkeypatch
):
    # With both standard variables set, the traces-specific one must win.
    # That ordering lives in the session's exporter factory, so the hooks
    # must not pass the generic value as an explicit protocol.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")

    exporting.create_session()

    assert captured_session_kwargs["otel_exporter"] == "otlp"
    assert captured_session_kwargs["otlp_protocol"] is None


def test_trulens_protocol_variable_is_passed_through(
    captured_session_kwargs, monkeypatch
):
    monkeypatch.setenv("TRULENS_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

    exporting.create_session()

    assert captured_session_kwargs["otlp_protocol"] == "http/protobuf"

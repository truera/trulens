import os


def is_otel_tracing_enabled() -> bool:
    return os.getenv("TRULENS_OTEL_TRACING", "").lower() in ["1", "true"]

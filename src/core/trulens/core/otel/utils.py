import os


def _is_env_var_disabled(var_name: str) -> bool:
    """Check if an environment variable is explicitly set to disable a feature.

    Returns True if the environment variable is explicitly set to "0" or "false"
    (case-insensitive), indicating the feature should be disabled.
    Returns False otherwise (feature enabled by default).
    """
    return os.getenv(var_name, "").lower() in ["0", "false"]


def is_otel_tracing_enabled() -> bool:
    """Check if OpenTelemetry tracing is enabled.

    Returns True by default unless TRULENS_OTEL_TRACING is explicitly set to "0" or "false".
    """
    return not _is_env_var_disabled("TRULENS_OTEL_TRACING")


def is_otel_backwards_compatibility_enabled() -> bool:
    """Check if OpenTelemetry backwards compatibility is enabled.

    Returns True by default unless TRULENS_OTEL_BACKWARDS_COMPATIBILITY is explicitly set to "0" or "false".
    """
    return not _is_env_var_disabled("TRULENS_OTEL_BACKWARDS_COMPATIBILITY")

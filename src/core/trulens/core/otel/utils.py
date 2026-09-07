import logging
import os

logger = logging.getLogger(__name__)

_OTEL_DISABLED_WARNING_EMITTED = False


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

    Warns once per process when tracing has been disabled, since the symptom is
    an absence of spans rather than an error.
    """
    global _OTEL_DISABLED_WARNING_EMITTED

    if _is_env_var_disabled("TRULENS_OTEL_TRACING"):
        if not _OTEL_DISABLED_WARNING_EMITTED:
            _OTEL_DISABLED_WARNING_EMITTED = True
            logger.warning(
                "OTEL tracing is DISABLED because TRULENS_OTEL_TRACING is set to"
                " %r. No spans will be recorded, and evaluations that read spans"
                " will find nothing. OTEL tracing is enabled by default; unset"
                " TRULENS_OTEL_TRACING to restore it.",
                os.getenv("TRULENS_OTEL_TRACING"),
            )
        return False

    return True


def is_otel_backwards_compatibility_enabled() -> bool:
    """Check if OpenTelemetry backwards compatibility is enabled.

    Returns True by default unless TRULENS_OTEL_BACKWARDS_COMPATIBILITY is explicitly set to "0" or "false".
    """
    return not _is_env_var_disabled("TRULENS_OTEL_BACKWARDS_COMPATIBILITY")

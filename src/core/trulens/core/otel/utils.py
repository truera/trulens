import os


def _is_env_var_set(var_name: str) -> bool:
    return os.getenv(var_name, "").lower() in ["1", "true"]


def is_otel_tracing_enabled() -> bool:
    return _is_env_var_set("TRULENS_OTEL_TRACING")


def is_otel_backwards_compatibility_enabled() -> bool:
    return _is_env_var_set("TRULENS_OTEL_BACKWARDS_COMPATIBILITY")

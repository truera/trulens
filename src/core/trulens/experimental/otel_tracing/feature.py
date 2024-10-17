"""Utilities for managing optional requirements of the experimental otel_tracing
feature."""

from trulens.core._utils import optional as optional_utils
from trulens.core.utils import imports as import_utils

with import_utils.OptionalImports(optional_utils.REQUIREMENT_OTEL) as oi:
    import opentelemetry


def assert_optionals_installed():
    oi.assert_installed(opentelemetry)


def are_optionals_installed():
    return not import_utils.is_dummy(opentelemetry)

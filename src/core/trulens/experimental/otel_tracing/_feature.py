"""Utilities for managing optional requirements of the experimental otel_tracing
feature."""

from trulens.core import experimental
from trulens.core.utils import imports as import_utils

FEATURE = experimental.Feature.OTEL_TRACING
"""Feature controlling the use of this module."""

REQUIREMENT = import_utils.format_import_errors(
    ["opentelemetry-api", "opentelemetry-sdk", "opentelemetry-proto"],
    purpose="otel_tracing experimental feature",
)
"""Optional modules required for the otel_tracing experimental feature."""

with import_utils.OptionalImports(REQUIREMENT) as oi:
    from opentelemetry import sdk
    from opentelemetry import trace


class _FeatureSetup(experimental._FeatureSetup):
    """Utilities for managing the otel_tracing experimental feature."""

    FEATURE = FEATURE
    REQUIREMENT = REQUIREMENT

    @staticmethod
    def assert_optionals_installed():
        """Asserts that the optional requirements for the otel_tracing feature are
        installed."""
        oi.assert_installed([sdk, trace])

    @staticmethod
    def are_optionals_installed():
        """Checks if the optional requirements for the otel_tracing feature are
        installed."""
        return not any(import_utils.is_dummy(m) for m in [sdk, trace])

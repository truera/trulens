from importlib.metadata import version
import sys

# Re-export key symbols so callers don't need to know the internal trace module.
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

# Explicit public API surface for this package.
__all__ = [
    "GenAIAttributes",
    "ResourceAttributes",
    "SpanAttributes",
]


def safe_importlib_package_name(package_name: str) -> str:
    """Convert a package name that may have periods in it to one that uses
    hyphens for periods but only if the python version is old.
    Copied from trulens-core to avoid a circular dependency."""

    return (
        package_name
        if sys.version_info >= (3, 10)
        else package_name.replace(".", "-")
    )


__version__ = version(safe_importlib_package_name(__package__ or __name__))

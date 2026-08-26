"""TruLens instrumentation for coding-agent client hooks."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from trulens.core.otel.client_hooks.models import HookEvent
from trulens.core.otel.client_hooks.service import HookService

try:
    __version__ = version("trulens-core")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["HookEvent", "HookService", "__version__"]

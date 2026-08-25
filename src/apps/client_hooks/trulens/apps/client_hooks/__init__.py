"""TruLens instrumentation for client-side coding agent hooks."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from trulens.apps.client_hooks.models import HookEvent
from trulens.apps.client_hooks.service import HookService

try:
    __version__ = version("trulens-apps-client-hooks")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["HookEvent", "HookService", "__version__"]

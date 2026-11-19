"""TypeSafe Jev provider for TruLens."""

# WARNING: This file does not follow the no-init aliases import standard.

from importlib.metadata import version

from trulens.core.utils import imports as import_utils
from trulens.providers.typesafe.provider import TypeSafe
from trulens.providers.typesafe.provider import TypeSafeDecision

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)

__all__ = ["TypeSafe", "TypeSafeDecision"]

"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-litellm` package installed.

    ```bash
    pip install trulens-providers-litellm
    ```
"""

# Suppress third-party library warnings before importing anything else
import logging
import warnings

# Suppress pkg_resources deprecation warning from munch and other libraries
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

# Suppress python-dotenv parsing warnings for malformed .env files
warnings.filterwarnings(
    "ignore",
    message="python-dotenv could not parse statement",
    category=UserWarning,
)

# Suppress python-dotenv logger warnings for malformed .env files
logging.getLogger("dotenv.main").setLevel(logging.ERROR)

from importlib.metadata import version  # noqa: E402

from trulens.core.utils.imports import safe_importlib_package_name  # noqa: E402
from trulens.providers.litellm.provider import LiteLLM  # noqa: E402

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "LiteLLM",
]

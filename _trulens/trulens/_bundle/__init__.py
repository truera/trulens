"""
TruLens: Don't just vibe check your LLM app!

"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))

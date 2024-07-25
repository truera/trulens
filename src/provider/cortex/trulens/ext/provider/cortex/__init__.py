import importlib.metadata

from trulens.ext.provider.cortex.provider import Cortex

__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    "Cortex",
]

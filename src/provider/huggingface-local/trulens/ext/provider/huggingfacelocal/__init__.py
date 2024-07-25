import importlib.metadata

from trulens.ext.provider.huggingfacelocal.provider import HuggingfaceLocal

__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = ["HuggingfaceLocal"]

import importlib.metadata

from trulens.ext.provider.bedrock.provider import Bedrock

__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    "Bedrock",
]

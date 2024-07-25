import importlib.metadata

from trulens.ext.provider.openai.provider import AzureOpenAI
from trulens.ext.provider.openai.provider import OpenAI

__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = ["OpenAI", "AzureOpenAI"]

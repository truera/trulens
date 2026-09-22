from functools import lru_cache

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from .config import Settings


@lru_cache(maxsize=None)
def _get_client(api_key: str) -> OpenAI:
    """Reuse a single OpenAI client per API key."""
    return OpenAI(api_key=api_key or None)


class OpenAIEmbeddings(Embeddings):
    """Embeddings via OpenAI (e.g. text-embedding-3-small)."""

    def __init__(self, model_name: str, api_key: str = "") -> None:
        self._model = model_name
        self._client = _get_client(api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model, input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(settings.embedding_model, settings.openai_api_key)

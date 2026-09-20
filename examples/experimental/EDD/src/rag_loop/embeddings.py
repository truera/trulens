from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from .config import Settings


class FastEmbedEmbeddings(Embeddings):
    """Local, keyless embeddings via FastEmbed (ONNX) — no OpenAI/Gemini call needed."""

    def __init__(self, model_name: str) -> None:
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(settings: Settings) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(settings.embedding_model)

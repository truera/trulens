import hashlib
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
import numpy as np


class MockEmbedding(BaseEmbedding):
    embed_dim: int

    def __init__(self, embed_dim: int, **kwargs: Any) -> None:
        super().__init__(embed_dim=embed_dim, **kwargs)

    def _convert_text_to_seed(self, text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)

    def _construct_vector_from_text(self, text: str) -> List[float]:
        np.random.seed(self._convert_text_to_seed(text))
        return np.random.random(size=self.embed_dim).tolist()

    @classmethod
    def class_name(cls) -> str:
        return "MockEmbedding"

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._construct_vector_from_text(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._construct_vector_from_text(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._construct_vector_from_text(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._construct_vector_from_text(query)

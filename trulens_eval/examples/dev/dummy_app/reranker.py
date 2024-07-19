from typing import List, Optional, Tuple

from examples.dev.dummy_app.dummy import Dummy

from trulens_eval.tru_custom_app import instrument


class DummyReranker(Dummy):
    """Dummy reranker that uses a text length to score chunks."""

    def __init__(self, *args, top_n: int = 2, **kwargs):
        super().__init__(*args, **kwargs)

        self.top_n = top_n
        self.model_name = "herpderp-v1-reranker"

    @instrument
    def rerank(
        self,
        query_text: str,
        chunks: List[str],
        chunk_scores: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """Fake chunk reranker."""

        # Pretend to allocate some data.
        self.dummy_allocate()

        # Fake delay.
        self.dummy_wait()

        chunks_and_scores = [
            (chunk, float(abs(len(chunk) - len(query_text))))
            for chunk in chunks
        ]

        return sorted(chunks_and_scores, key=lambda cs: cs[1])[:self.top_n]

    @instrument
    async def arerank(
        self,
        query_text: str,
        chunks: List[str],
        chunk_scores: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """Fake chunk reranker."""

        # Pretend to allocate some data.
        self.dummy_allocate()

        # Fake delay.
        await self.dummy_await()

        chunks_and_scores = [
            (chunk, float(abs(len(chunk) - len(query_text))))
            for chunk in chunks
        ]

        return sorted(chunks_and_scores, key=lambda cs: cs[1])[:self.top_n]

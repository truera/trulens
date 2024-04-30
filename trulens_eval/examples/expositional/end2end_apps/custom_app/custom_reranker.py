from typing import List, Optional, Tuple

from examples.expositional.end2end_apps.custom_app.dummy import Dummy

from trulens_eval.schema import record as mod_record_schema
from trulens_eval.trace import span as mod_span
from trulens_eval.tru_custom_app import instrument


class CustomReranker(Dummy):
    """Fake reranker that uses a text length to score chunks."""

    def __init__(self, *args, top_n: int = 2, **kwargs):
        super().__init__(*args, **kwargs)

        self.top_n = top_n
        self.model_name = "herpderp-v1-reranker"

    @instrument
    def rerank(self, query_text: str, chunks: List[str], chunk_scores: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """Fake chunk reranker."""

        # Pretend to allocate some data.
        self.dummy_allocate()

        # Fake delay.
        self.dummy_wait()

        chunks_and_scores = [(chunk, float(abs(len(chunk) - len(query_text)))) for chunk in chunks]

        return sorted(chunks_and_scores, key=lambda cs: cs[1])[:self.top_n]

    @rerank.is_span(
        span_type=mod_span.SpanReranker
    )
    def update_span(
        self,
        call: mod_record_schema.RecordAppCall,
        span: mod_span.SpanReranker
    ):
        """Fill in reranking span info based on what the dummy reranker did."""

        span.model_name = self.model_name
        span.top_n = self.top_n

        span.query_text = call.args['query_text']

        span.input_context_texts = call.args['chunks']

        span.input_context_scores = call.args['chunk_scores']

        output_chunks_and_scores = call.rets
        output_context_texts = [cs[0] for cs in output_chunks_and_scores]

        span.output_ranks = [call.args['chunks'].index(chunk) for chunk in output_context_texts]
    
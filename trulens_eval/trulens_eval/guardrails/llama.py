from concurrent.futures import wait
from typing import List

from trulens_eval.feedback import Feedback
from trulens_eval.utils.containers import first
from trulens_eval.utils.containers import second
from trulens_eval.utils.threading import ThreadPoolExecutor

from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LLAMA

with OptionalImports(messages=REQUIREMENT_LLAMA):
    import llama_index
    from llama_index.core.indices.vector_store.retrievers.retriever import \
        VectorIndexRetriever
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.schema import NodeWithScore


class WithFeedbackFilterNodes(VectorIndexRetriever):
    feedback: Feedback
    threshold: float

    def __init__(self, feedback: Feedback, threshold: float, *args, **kwargs):
        """
        A VectorIndexRetriever that filters documents using a minimum threshold
        on a feedback function before returning them.

        - feedback: Feedback - use this feedback function to score each
        document.
        
        - threshold: float - and keep documents only if their feedback value is
        at least this threshold.
        """

        super().__init__(*args, **kwargs)

        self.feedback = feedback
        self.threshold = threshold

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Get relevant docs using super class:
        nodes = super()._retrieve(query_bundle)

        ex = ThreadPoolExecutor(max_workers=max(1, len(nodes)))

        # Evaluate the filter on each, in parallel.
        futures = (
            (
                node,
                ex.submit(
                    lambda query, node: self.feedback(
                        query.query_str, node.node.get_text()
                    ) > self.threshold,
                    query=query_bundle,
                    node=node
                )
            ) for node in nodes
        )

        wait([future for (_, future) in futures])

        results = ((node, future.result()) for (node, future) in futures)
        filtered = map(first, filter(second, results))

        # Return only the filtered ones.
        return list(filtered)

    @staticmethod
    def of_index_retriever(retriever: VectorIndexRetriever, **kwargs):
        return WithFeedbackFilterNodes(
            index=retriever._index,
            similarty_top_k=retriever._similarity_top_k,
            vectore_store_query_mode=retriever._vector_store_query_mode,
            filters=retriever._filters,
            alpha=retriever._alpha,
            doc_ids=retriever._doc_ids,
            **kwargs
        )

from concurrent.futures import wait
from typing import List

from trulens_eval.feedback import Feedback
from trulens_eval.utils.containers import first
from trulens_eval.utils.containers import second
from trulens_eval.utils.threading import ThreadPoolExecutor
from trulens_eval.utils.serial import model_dump

from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LLAMA

with OptionalImports(messages=REQUIREMENT_LLAMA):
    import llama_index
    from llama_index.core.query_engine.retriever_query_engine import \
        RetrieverQueryEngine
    from llama_index.core.indices.vector_store.base import VectorStoreIndex
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.schema import NodeWithScore

class WithFeedbackFilterNodes(RetrieverQueryEngine):
    feedback: Feedback
    threshold: float

    def __init__(self, query_engine: RetrieverQueryEngine, feedback: Feedback, threshold: float, *args, **kwargs):
        """
        A BaseQueryEngine that filters documents using a minimum threshold
        on a feedback function before returning them.

        - feedback: Feedback - use this feedback function to score each
        document.
        
        - threshold: float - and keep documents only if their feedback value is
        at least this threshold.
        """
        self.query_engine = query_engine
        self.feedback = feedback
        self.threshold = threshold

    def query(self, query: QueryBundle, **kwargs) -> List[NodeWithScore]:
        # Get relevant docs using super class:
        nodes = self.query_engine.retrieve(query_bundle=query)
        ex = ThreadPoolExecutor(max_workers=max(1, len(nodes)))

        # Evaluate the filter on each, in parallel.
        futures = list(
            (
                node,
                ex.submit(
                    (
                        lambda node: self.feedback(
                            query, node.node.get_text()
                        ) > self.threshold
                    ),
                    node=node
                )
            ) for node in nodes
        )

        wait([future for (_, future) in futures])

        results = list((node, future.result()) for (node, future) in futures)
        filtered = map(first, filter(second, results))

        filtered_nodes = list(filtered)
        return self.query_engine.synthesize(query_bundle=query, nodes=filtered_nodes, **kwargs)

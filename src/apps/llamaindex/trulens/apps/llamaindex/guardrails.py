from concurrent.futures import as_completed
from typing import List

from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.schema import NodeWithScore
from trulens.core import Feedback
from trulens.core.utils.threading import ThreadPoolExecutor


class WithFeedbackFilterNodes(RetrieverQueryEngine):
    feedback: Feedback
    threshold: float
    """
    A BaseQueryEngine that filters documents using a minimum threshold
    on a feedback function before returning them.

    Args:
        feedback (Feedback): use this feedback function to score each document.
        threshold (float): and keep documents only if their feedback value is at least this threshold.

    Example: "Using TruLens guardrail context filters with Llama-Index"
        ```python
        from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes

        # note: feedback function used for guardrail must only return a score, not also reasons
        feedback = (
            Feedback(provider.context_relevance)
            .on_input()
            .on(context)
        )

        filtered_query_engine = WithFeedbackFilterNodes(query_engine, feedback=feedback, threshold=0.5)

        tru_recorder = TruLlama(filtered_query_engine,
            app_name="LlamaIndex_App",
            app_version="v1_filtered"
        )

        with tru_recorder as recording:
            llm_response = filtered_query_engine.query("What did the author do growing up?")
        ```
    """

    def __init__(
        self,
        query_engine: RetrieverQueryEngine,
        feedback: Feedback,
        threshold: float,
        *args,
        **kwargs,
    ):
        self.query_engine = query_engine
        self.feedback = feedback
        self.threshold = threshold

    def query(self, query: QueryBundle, **kwargs) -> List[NodeWithScore]:
        """
        An extended query method that will:

        1. Query the engine with the given query bundle (like before).
        2. Evaluate nodes with a specified feedback function.
        3. Filter out nodes that do not meet the minimum threshold.
        4. Synthesize with only the filtered nodes.

        Args:
            query (QueryBundle): The query bundle to search for relevant nodes.
            **kwargs: additional keyword arguments.

        Returns:
            List[NodeWithScore]: a list of filtered, relevant nodes.
        """
        # Get relevant docs using super class:
        nodes = self.query_engine.retrieve(query_bundle=query)

        with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as ex:
            future_to_node = {
                ex.submit(
                    lambda node=node: self.feedback(query, node.node.get_text())
                ): node
                for node in nodes
            }
            filtered = []
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                result = future.result()
                if not isinstance(result, float):
                    raise ValueError(
                        "Guardrails can only be used with feedback functions that return a float."
                    )
                if (
                    self.feedback.higher_is_better and result > self.threshold
                ) or (
                    not self.feedback.higher_is_better
                    and result < self.threshold
                ):
                    filtered.append(node)

        filtered_nodes = list(filtered)
        return self.query_engine.synthesize(
            query_bundle=query, nodes=filtered_nodes, **kwargs
        )

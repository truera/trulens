from typing import Callable, Iterable, List

from llama_index.data_structs.node import NodeType
from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from pydantic import Field

from trulens_eval import Feedback
from trulens_eval.tru_feedback import Feedback
from trulens_eval.tru_model import COMPONENT_CATEGORY
from trulens_eval.util import Class
from trulens_eval.util import first
from trulens_eval.util import second
from trulens_eval.util import TP


class Is:
    """
    Various checks for typical llama index components based on their names (i.e.
    without actually loading them). See util.py:WithClassInfo for more.
    """

    @staticmethod
    def engine(cls: Class):
        return cls.noserio_issubclass(
            module_name="llama_index.indices.query.base",
            class_name="BaseQueryEngine"
        )

    @staticmethod
    def retriever(cls: Class):
        return cls.noserio_issubclass(
            module_name="llama_index.indices.base_retriever",
            class_name="BaseRetriever"
        )

    @staticmethod
    def selector(cls: Class):
        return cls.noserio_issubclass(
            module_name="llama_index.selectors.types",
            class_name="BaseSelector"
        )

    @staticmethod
    def what(cls: Class) -> Iterable[COMPONENT_CATEGORY]:
        CHECKERS = [Is.engine, Is.retriever, Is.selector]

        for checker in CHECKERS:
            if checker(cls):
                yield checker.__name__


class WithFeedbackFilterNodes(VectorIndexRetriever):
    feedback: Feedback
    threshold: float

    def __init__(self, feedback: Feedback, threshold: float, *args, **kwargs):
        """
        A VectorStoreRetriever that filters documents using a minimum threshold
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

        # Evaluate the filter on each, in parallel.
        promises = (
            (
                node, TP().promise(
                    lambda query, node: self.feedback(
                        query.query_str, node.node.get_text()
                    ) > self.threshold,
                    query=query_bundle,
                    node=node
                )
            ) for node in nodes
        )
        results = ((node, promise.get()) for (node, promise) in promises)
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

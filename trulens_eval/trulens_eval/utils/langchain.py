from typing import Callable, List

from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import Field
from trulens_eval.tru_feedback import Feedback

from trulens_eval.util import TP, first, second


class WithFeedbackFilterDocuments(VectorStoreRetriever):
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

        super().__init__(
            feedback=feedback, threshold=threshold, *args, **kwargs
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Get relevant docs using super class:
        docs = super().get_relevant_documents(query)

        # Evaluate the filter on each, in parallel.
        promises = (
            (
                doc, TP().promise(
                    lambda doc, query: self.feedback(query, doc.page_content) >
                    self.threshold,
                    query=query,
                    doc=doc
                )
            ) for doc in docs
        )
        results = ((doc, promise.get()) for (doc, promise) in promises)
        filtered = map(first, filter(second, results))

        # Return only the filtered ones.
        return list(filtered)

    @staticmethod
    def of_retriever(retriever: VectorStoreRetriever, **kwargs):
        return WithFeedbackFilterDocuments(**kwargs, **retriever.dict())

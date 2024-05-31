"""
Utilities for langchain apps. Includes component categories that organize
various langchain classes and example classes:

- `WithFeedbackFilterDocuments`: a `VectorStoreRetriever` that filters retrieved
  documents via a threshold on a specified feedback function.
"""

from concurrent.futures import wait
from typing import List, Type

from trulens_eval import app
from trulens_eval.feedback import Feedback
from trulens_eval.utils.containers import first
from trulens_eval.utils.containers import second
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LANGCHAIN
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import model_dump
from trulens_eval.utils.threading import ThreadPoolExecutor

with OptionalImports(messages=REQUIREMENT_LANGCHAIN):
    import langchain
    from langchain.schema import Document
    from langchain.vectorstores.base import VectorStoreRetriever


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
            *args, feedback=feedback, threshold=threshold, **kwargs
        )

    # Signature must match
    # langchain.schema.retriever.BaseRetriever._get_relevant_documents .
    def _get_relevant_documents(self, query: str, *,
                                run_manager) -> List[Document]:
        # Get relevant docs using super class:
        docs = super()._get_relevant_documents(query, run_manager=run_manager)

        # Evaluate the filter on each, in parallel.
        ex = ThreadPoolExecutor(max_workers=max(1, len(docs)))

        futures = list(
            (
                doc,
                ex.submit(
                    (
                        lambda doc, query: self.
                        feedback(query, doc.page_content) > self.threshold
                    ),
                    query=query,
                    doc=doc
                )
            ) for doc in docs
        )

        wait([future for (_, future) in futures])

        results = list((doc, future.result()) for (doc, future) in futures)
        filtered = map(first, filter(second, results))

        # Return only the filtered ones.
        return list(filtered)

    @staticmethod
    def of_retriever(retriever: VectorStoreRetriever, **kwargs):
        return WithFeedbackFilterDocuments(**kwargs, **(model_dump(retriever)))

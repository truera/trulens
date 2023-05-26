from typing import Callable, List

from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import Field

from trulens_eval.util import TP, first, second


class WithFilterDocuments(VectorStoreRetriever):
    filter_func: Callable = Field(exclude=True)

    def __init__(
        self, filter_func: Callable[[Document], bool], *args, **kwargs
    ):
        """
        A VectorStoreRetriever that filters documents before returning them.

        - filter_func: Callable[[Document], bool] - apply this filter before
          returning documents. Will return only documents for which the filter
          returns true.
        """

        super().__init__(filter_func=filter_func, *args, **kwargs)

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Get relevant docs using super class:
        docs = super().get_relevant_documents(query)

        # Evaluate the filter on each, in parallel.
        promises = (
            (doc, TP().promise(self.filter_func, query=query, doc=doc))
            for doc in docs
        )
        results = ((doc, promise.get()) for (doc, promise) in promises)
        filtered = map(first, filter(second, results))

        # Return only the filtered ones.
        return list(filtered)

    @staticmethod
    def of_retriever(retriever: VectorStoreRetriever, filter_func: Callable):
        return WithFilterDocuments(filter_func=filter_func, **retriever.dict())
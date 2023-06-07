from typing import Callable, Iterable, List

import langchain
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import Field
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import Class
from trulens_eval.util import TP, first, second

CLASSES_TO_INSTRUMENT = {
    langchain.chains.base.Chain,
    langchain.vectorstores.base.BaseRetriever,
    langchain.schema.BaseRetriever,
    langchain.llms.base.BaseLLM,
    langchain.prompts.base.BasePromptTemplate,
    langchain.schema.BaseMemory,
    langchain.schema.BaseChatMessageHistory
}

# Instrument only methods with these names and of these classes.
METHODS_TO_INSTRUMENT = {
    "_call": lambda o: isinstance(o, langchain.chains.base.Chain),
    "get_relevant_documents": lambda o: True,  # VectorStoreRetriever
    "__call__": lambda o: isinstance(o, Feedback)  # Feedback
}

class Is:
    """
    Various checks for typical langchain components based on their names (i.e.
    without actually loading them). See util.py:WithClassInfo for more.
    """

    @staticmethod
    def chain(cls: Class):
        return cls.noserio_issubclass(module_name="langchain.chains.base", class_name="Chain")

    @staticmethod
    def vector_store(cls: Class):
        return cls.noserio_issubclass(module_name="langchain.vectorstores", class_name="VectorStoreRetriever")

    @staticmethod
    def retriever(cls: Class):
        return cls.noserio_issubclass(module_name="langchain.schema", class_name="BaseRetriever")

    @staticmethod
    def llm(cls: Class):
        return cls.noserio_issubclass(module_name="langchain.llms.base", class_name="BaseLLM")

    @staticmethod
    def prompt(cls: Class):
        return cls.noserio_issubclass(module_name="langchain.prompts.base", class_name="BasePromptTemplate")

    @staticmethod
    def memory(cls: Class):
        return cls.noserio_issubclass(module_name="langchain.schema", class_name="BaseMemory")

    @staticmethod
    def chathistory(cls: Class):
        return cls.noserio_issubclass(module_name="langchain.schema", class_name="BaseChatMessageHistory")
    
    @staticmethod
    def what(cls: Class) -> Iterable[str]:
        CHECKERS = [Is.chain, Is.vector_store, Is.retriever, Is.llm, Is.prompt, Is.memory, Is.chathistory]

        for checker in CHECKERS:
            if checker(cls):
                yield checker.__name__


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

from concurrent.futures import wait
from typing import List

from trulens_eval.feedback import Feedback
from trulens_eval.utils.containers import first
from trulens_eval.utils.containers import second
from trulens_eval.utils.serial import model_dump
from trulens_eval.utils.threading import ThreadPoolExecutor

from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LANGCHAIN


with OptionalImports(messages=REQUIREMENT_LANGCHAIN):
    import langchain
    from langchain.schema import Document
    from langchain.vectorstores.base import VectorStoreRetriever


class WithFeedbackFilterDocuments(VectorStoreRetriever):
    feedback: Feedback
    threshold: float
    """
    A VectorStoreRetriever that filters documents using a minimum threshold
    on a feedback function before returning them.
    
    Args:
        feedback (Feedback): use this feedback function to score each document.

        threshold (float): and keep documents only if their feedback value is at least this threshold.

    !!! example "Using TruLens guardrail context filters with Langchain"

        ```python
        from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments

        # note: feedback function used for guardrail must only return a score, not also reasons
        feedback = Feedback(provider.context_relevance).on_input().on(context)

        filtered_retriever = WithFeedbackFilterDocuments.of_retriever(
            retriever=retriever,
            feedback=feedback,
            threshold=0.5
        )

        rag_chain = {"context": filtered_retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

        tru_recorder = TruChain(rag_chain,
            app_id='Chain1_ChatApplication_Filtered')

        with tru_recorder as recording:
            llm_response = rag_chain.invoke("What is Task Decomposition?")
        ```
    """

    def __init__(self, feedback: Feedback, threshold: float, *args, **kwargs):
        super().__init__(
            *args, feedback=feedback, threshold=threshold, **kwargs
        )

    # Signature must match
    # langchain.schema.retriever.BaseRetriever._get_relevant_documents .
    def _get_relevant_documents(self, query: str, *,
                                run_manager) -> List[Document]:
        """
        An internal method to accomplish three tasks:

        1. Get relevant documents.
        2. Evaluate documents with a specified feedback function.
        3. Filter out documents that do not meet the minimum threshold.

        Args:
            query: str - the query string to search for relevant documents.
            
            run_manager: RunManager - the run manager to handle document retrieval.

        Returns:
        - List[Document]: a list of filtered, relevant documents.
        """
        # Get relevant docs using super class:
        docs = super()._get_relevant_documents(query, run_manager=run_manager)

        # Evaluate the filter on each, in parallel.
        ex = ThreadPoolExecutor(max_workers=max(1, len(docs)))

        futures = list(
            (
                doc,
                ex.submit(
                    lambda doc, query: self.feedback(query, doc.page_content),
                    doc,
                    query
                )
            ) for doc in docs
        )
        wait([future for (_, future) in futures])

        results = list((doc, future.result()) for (doc, future) in futures)
        for doc, result in results:
            if not isinstance(result, float):
                raise ValueError("Guardrails can only be used with feedback functions that return a float.")
        if self.feedback.higher_is_better:
            filtered = map(first, filter(lambda x: second(x) > self.threshold, results))
        else:
            filtered = map(first, filter(lambda x: second(x) < self.threshold, results))

        # Return only the filtered ones.
        return list(filtered)

    @staticmethod
    def of_retriever(retriever: VectorStoreRetriever, **kwargs):
        """
        Create a new instance of WithFeedbackFilterDocuments based on an existing retriever.

        The new instance will:

        1. Get relevant documents (like the existing retriever its based on).
        2. Evaluate documents with a specified feedback function.
        3. Filter out documents that do not meet the minimum threshold.

        Args:
            retriever: VectorStoreRetriever - the base retriever to use.

            **kwargs: additional keyword arguments.

        Returns:
        - WithFeedbackFilterDocuments: a new instance of WithFeedbackFilterDocuments.
        """
        return WithFeedbackFilterDocuments(**kwargs, **(model_dump(retriever)))

from concurrent.futures import wait
from typing import List

from trulens_eval.feedback import Feedback
from trulens_eval.utils.containers import first
from trulens_eval.utils.containers import second
from trulens_eval.utils.serial import model_dump
from trulens_eval.utils.threading import ThreadPoolExecutor

from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LANGCHAIN
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.serial import JSON


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

        !!! example "Using TruLens guardrail context filters with Langchain"

            ```python
            from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments

            # note: feedback function used for guardrail must only return a score, not also reasons
            f_context_relevance_score = (
                Feedback(provider.context_relevance)
                .on_input()
                .on(context)
                .aggregate(np.mean)
            )

            filtered_retriever = WithFeedbackFilterDocuments.of_retriever(
                    retriever=retriever,
                    feedback=f_context_relevance_score,
                    threshold=0.5
                )

            rag_chain = (
                {"context": filtered_retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            tru_recorder = TruChain(rag_chain,
                app_id='Chain1_ChatApplication_Filtered',
                feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness])

            with tru_recorder as recording:
                llm_response = rag_chain.invoke("What is Task Decomposition?")
            ```  
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

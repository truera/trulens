from concurrent.futures import as_completed

from trulens_eval.feedback import Feedback
from trulens_eval.utils.threading import ThreadPoolExecutor


class context_filter:
    """
    Provides a decorator to filter contexts based on a given feedback and threshold.

    Parameters:
    feedback (Feedback): The feedback object to use for filtering.
    threshold (float): The minimum feedback value required for a context to be included.
    keyword_for_prompt (str): Keyword argument to decorator to use for prompt.

    !!! example

        ```python
        feedback = Feedback(provider.context_relevance, name="Context Relevance")
        class RAG_from_scratch:
            ...
            @context_filter(feedback, 0.5, "query")
            def retrieve(self, *, query: str) -> list:
                results = vector_store.query(
                    query_texts=query,
                    n_results=3
                )
                return [doc for sublist in results['documents'] for doc in sublist]
            ...
        ```
    """

    def __init__(
        self, feedback: Feedback, threshold: float, keyword_for_prompt: str
    ):
        self.feedback = feedback
        self.threshold = threshold
        self.keyword_for_prompt = keyword_for_prompt

    def __call__(self, func):

        def wrapper(*args, **kwargs):
            contexts = func(*args, **kwargs)
            with ThreadPoolExecutor(max_workers=max(1, len(contexts))) as ex:
                future_to_context = {
                    ex.submit(
                        lambda context=context: self.
                        feedback(kwargs[self.keyword_for_prompt], context)
                    ): context for context in contexts
                }
                filtered = []
                for future in as_completed(future_to_context):
                    context = future_to_context[future]
                    result = future.result()
                    if not isinstance(result, float):
                        raise ValueError(
                            "Guardrails can only be used with feedback functions that return a float."
                        )
                    if (self.feedback.higher_is_better and result > self.threshold) or \
                       (not self.feedback.higher_is_better and result < self.threshold):
                        filtered.append(context)
                return filtered

        # note: the following information is manually written to the wrapper because @functools.wraps(func) causes breaking of the method.
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

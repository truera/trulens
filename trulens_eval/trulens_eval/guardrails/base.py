from concurrent.futures import wait
from trulens_eval.feedback import Feedback
from trulens_eval.utils.containers import first
from trulens_eval.utils.containers import second
from trulens_eval.utils.threading import ThreadPoolExecutor
from functools import wraps

class context_filter:
    def __init__(self, feedback: Feedback, threshold: float):
        self.feedback = feedback
        self.threshold = threshold
    """
    ContextFilter is a class that filters contexts based on a given feedback and threshold.

    Parameters:
    feedback (Feedback): The feedback object to use for filtering.
    threshold (float): The minimum feedback value required for a context to be included.

    !!! example

        ```python
        feedback = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        @context_filter(feedback, 0.5)
        def retrieve(query: str) -> list:
            results = vector_store.query(
            query_texts=query,
            n_results=3
        )
        return [doc for sublist in results['documents'] for doc in sublist]
        ```
    """

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            contexts = func(*args, **kwargs)
            ex = ThreadPoolExecutor(max_workers=max(1, len(contexts)))
            futures = list(
                (
                    context,
                    ex.submit(
                        lambda context=context: self.feedback(
                            args[1], context
                        )
                    )
                ) for context in contexts
            )
            wait([future for (_, future) in futures])
            results = list((context, future.result()) for (context, future) in futures)
            for context, result in results:
                if not isinstance(result, float):
                    raise ValueError("Guardrails can only be used with feedback functions that return a float.")
            filtered = map(first, filter(lambda x: second(x) > self.threshold, results))
            return list(filtered)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

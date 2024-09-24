from concurrent.futures import as_completed
import inspect
import logging
from typing import Optional

from trulens.core import Feedback
from trulens.core.utils.threading import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class context_filter:
    """
    Provides a decorator to filter contexts based on a given feedback and threshold.

    Args:
        feedback (Feedback): The feedback object to use for filtering.
        threshold (float): The minimum feedback value required for a context to be included.
        keyword_for_prompt (str): Keyword argument to decorator to use for prompt.

    Example:
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
        self,
        feedback: Feedback,
        threshold: float,
        keyword_for_prompt: Optional[str] = None,
    ):
        self.feedback = feedback
        self.threshold = threshold
        self.keyword_for_prompt = keyword_for_prompt

    def __call__(self, func):
        sig = inspect.signature(func)

        if self.keyword_for_prompt is not None:
            if self.keyword_for_prompt not in sig.parameters:
                raise TypeError(
                    f"Keyword argument '{self.keyword_for_prompt}' not found in `{func.__name__}` signature."
                )
        else:
            # For backwards compatibility, allow inference of keyword_for_prompt:
            first_arg = list(k for k in sig.parameters.keys() if k != "self")[0]
            self.keyword_for_prompt = first_arg
            logger.warn(
                f"Assuming `{self.keyword_for_prompt}` is the `{func.__name__}` arg to filter. "
                "Specify `keyword_for_prompt` to avoid this warning."
            )

        def wrapper(*args, **kwargs):
            bindings = sig.bind(*args, **kwargs)

            contexts = func(*args, **kwargs)
            with ThreadPoolExecutor(max_workers=max(1, len(contexts))) as ex:
                future_to_context = {
                    ex.submit(
                        lambda context=context: self.feedback(
                            bindings.arguments[self.keyword_for_prompt], context
                        )
                    ): context
                    for context in contexts
                }
                filtered = []
                for future in as_completed(future_to_context):
                    context = future_to_context[future]
                    result = future.result()
                    if not isinstance(result, float):
                        raise ValueError(
                            "Guardrails can only be used with feedback functions that return a float."
                        )
                    if (
                        self.feedback.higher_is_better
                        and result > self.threshold
                    ) or (
                        not self.feedback.higher_is_better
                        and result < self.threshold
                    ):
                        filtered.append(context)
                return filtered

        # note: the following information is manually written to the wrapper because @functools.wraps(func) causes breaking of the method.
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

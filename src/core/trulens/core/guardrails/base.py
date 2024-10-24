from concurrent.futures import as_completed
import inspect
import logging
from typing import Optional

from trulens.core.feedback import feedback as core_feedback
from trulens.core.utils import threading as threading_utils

logger = logging.getLogger(__name__)


class context_filter:
    """Provides a decorator to filter contexts based on a given feedback and threshold.

    Args:
        feedback: The feedback object to use for filtering.

        threshold: The minimum feedback value required for a context to be included.

        keyword_for_prompt: Keyword argument to decorator to use for prompt.

    Example:
        ```python
        from trulens.core.guardrails.base import context_filter

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
        feedback: core_feedback.Feedback,
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
            logger.warning(
                f"Assuming `{self.keyword_for_prompt}` is the `{func.__name__}` arg to filter. "
                "Specify `keyword_for_prompt` to avoid this warning."
            )

        def wrapper(*args, **kwargs):
            bindings = sig.bind(*args, **kwargs)

            contexts = func(*args, **kwargs)
            with threading_utils.ThreadPoolExecutor(
                max_workers=max(1, len(contexts))
            ) as ex:
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
                            "`context_filter` can only be used with feedback functions that return a float."
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
        wrapper.__signature__ = sig
        return wrapper


class block_input:
    """Provides a decorator to block input based on a given feedback and threshold.

    Args:
        feedback: The feedback object to use for blocking.
        threshold: The minimum feedback value required for a context to be included.
        keyword_for_prompt: Keyword argument to decorator to use for prompt.
        return_value: The value to return if the input is blocked. Defaults to None.

    Example:
        ```python
        from trulens.core.guardrails.base import block_input

        feedback = Feedback(provider.criminality, higher_is_better = False)

        class safe_input_chat_app:
            @instrument
            @block_input(feedback=feedback,
                threshold=0.9,
                keyword_for_prompt="question",
                return_value="I couldn't find an answer to your question.")
            def generate_completion(self, question: str) -> str:
                completion = (
                    oai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        messages=[
                            {
                                "role": "user",
                                "content": f"{question}",
                            }
                        ],
                    )
                    .choices[0]
                    .message.content
                )
                return completion
        ```
    """

    def __init__(
        self,
        feedback: core_feedback.Feedback,
        threshold: float,
        keyword_for_prompt: Optional[str] = None,
        return_value: Optional[str] = None,
    ):
        self.feedback = feedback
        self.threshold = threshold
        self.keyword_for_prompt = keyword_for_prompt
        self.return_value = return_value

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
            logger.warning(
                f"Assuming `{self.keyword_for_prompt}` is the `{func.__name__}` arg to block on. "
                "Specify `keyword_for_prompt` to avoid this warning."
            )

        def wrapper(*args, **kwargs):
            bindings = sig.bind(*args, **kwargs)
            keyword_value = bindings.arguments[self.keyword_for_prompt]
            result = self.feedback(keyword_value)
            if not isinstance(result, float):
                raise ValueError(
                    "`block_input` can only be used with feedback functions that return a float."
                )
            if (self.feedback.higher_is_better and result < self.threshold) or (
                not self.feedback.higher_is_better and result > self.threshold
            ):
                return self.return_value

            return func(*args, **kwargs)

        # note: the following information is manually written to the wrapper because @functools.wraps(func) causes breaking of the method.
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__signature__ = sig
        return wrapper


class block_output:
    """Provides a decorator to block output based on a given feedback and threshold.

    Args:
        feedback: The feedback object to use for blocking. It must only take a single argument.
        threshold: The minimum feedback value required for a context to be included.
        return_value: The value to return if the input is blocked. Defaults to None.

    Example:
        ```python
        from trulens.core.guardrails.base import block_output

        feedback = Feedback(provider.criminality, higher_is_better = False)

        class safe_output_chat_app:
            @instrument
            @block_output(feedback = feedback,
                threshold = 0.5,
                return_value = "Sorry, I couldn't find an answer to your question.")
            def chat(self, question: str) -> str:
                completion = (
                    oai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        messages=[
                            {
                                "role": "user",
                                "content": f"{question}",
                            }
                        ],
                    )
                    .choices[0]
                    .message.content
                )
                return completion
        ```
    """

    def __init__(
        self,
        feedback: core_feedback.Feedback,
        threshold: float,
        return_value: Optional[str] = None,
    ):
        self.feedback = feedback
        self.threshold = threshold
        self.return_value = return_value

    def __call__(self, func):
        sig = inspect.signature(func)

        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            result = self.feedback(output)
            if not isinstance(result, float):
                raise ValueError(
                    "`block_output` can only be used with feedback functions that return a float."
                )
            if (self.feedback.higher_is_better and result < self.threshold) or (
                not self.feedback.higher_is_better and result > self.threshold
            ):
                return self.return_value
            else:
                return output

        # note: the following information is manually written to the wrapper because @functools.wraps(func) causes breaking of the method.
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__signature__ = sig
        return wrapper

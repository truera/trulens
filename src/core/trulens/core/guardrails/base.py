from concurrent.futures import as_completed
from contextlib import contextmanager
import inspect
import logging
from typing import Optional

from opentelemetry import trace as otel_trace
from trulens.core.metric import metric as core_metric
from trulens.core.utils import threading as threading_utils
from trulens.experimental.otel_tracing.core.session import TRULENS_SERVICE_NAME
from trulens.experimental.otel_tracing.core.span import (
    set_general_span_attributes,
)
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


@contextmanager
def _guardrail_span(name: str, threshold: float):
    """Context manager that emits a GUARDRAIL span with standard attributes.

    Yields the span so callers can set SCORE and PASSED after computing them,
    ensuring the span duration covers the actual feedback evaluation work.
    """
    tracer = otel_trace.get_tracer_provider().get_tracer(TRULENS_SERVICE_NAME)
    with tracer.start_as_current_span(name) as span:
        set_general_span_attributes(span, SpanAttributes.SpanType.GUARDRAIL)
        span.set_attribute(SpanAttributes.GUARDRAIL.NAME, name)
        span.set_attribute(SpanAttributes.GUARDRAIL.THRESHOLD, threshold)
        yield span


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
        feedback: core_metric.Metric,
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
            guardrail_name = getattr(self.feedback, "name", func.__name__)

            def _evaluate_context(context) -> tuple:
                """Evaluate feedback for one context; returns (result, passed)."""
                result = self.feedback(
                    bindings.arguments[self.keyword_for_prompt], context
                )
                if not isinstance(result, float):
                    raise ValueError(
                        "`context_filter` can only be used with feedback functions that return a float."
                    )
                passed = (
                    self.feedback.higher_is_better and result > self.threshold
                ) or (
                    not self.feedback.higher_is_better
                    and result < self.threshold
                )
                return result, passed

            with threading_utils.ThreadPoolExecutor(
                max_workers=max(1, len(contexts))
            ) as ex:
                future_to_context = {
                    ex.submit(_evaluate_context, context): context
                    for context in contexts
                }
                filtered = []
                for future in as_completed(future_to_context):
                    context = future_to_context[future]
                    result, passed = future.result()
                    with _guardrail_span(
                        guardrail_name, self.threshold
                    ) as span:
                        span.set_attribute(
                            SpanAttributes.GUARDRAIL.SCORE, result
                        )
                        span.set_attribute(
                            SpanAttributes.GUARDRAIL.PASSED, passed
                        )
                    if passed:
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
            @instrument()
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
        feedback: core_metric.Metric,
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
            guardrail_name = getattr(self.feedback, "name", func.__name__)

            with _guardrail_span(guardrail_name, self.threshold) as span:
                result = self.feedback(keyword_value)
                if not isinstance(result, float):
                    raise ValueError(
                        "`block_input` can only be used with feedback functions that return a float."
                    )
                blocked = (
                    self.feedback.higher_is_better and result < self.threshold
                ) or (
                    not self.feedback.higher_is_better
                    and result > self.threshold
                )
                span.set_attribute(SpanAttributes.GUARDRAIL.SCORE, result)
                span.set_attribute(SpanAttributes.GUARDRAIL.PASSED, not blocked)

            if blocked:
                return self.return_value

            return func(*args, **kwargs)

        # note: the following information is manually written to the wrapper because @functools.wraps(func) causes breaking of the method.
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__signature__ = sig
        return wrapper


class block_output:
    """Provides a decorator to block output based on a given feedback and threshold.

    Also supports streaming (generator/async generator) functions: the
    feedback is periodically re-evaluated against the *accumulated* output
    so far as chunks arrive, every `check_every_n_chunks` chunks, so an
    unsafe response can be cut off partway through rather than only being
    caught (uselessly, since it would already be fully streamed to the
    caller by then) after the whole thing has been generated.

    !!! warning "Streaming guardrails can only block *future* chunks"
        Once a chunk has been yielded it can't be un-sent -- if a violation
        is detected at checkpoint N, chunks 1..N (whatever text they made
        up) have already reached the caller. Lower `check_every_n_chunks`
        catches problems earlier at the cost of more (typically
        LLM-judge-backed) feedback calls during the stream; there's no
        setting that guarantees zero unsafe output ever reaches the caller
        for a true token-by-token stream. A final check also runs once the
        stream ends (covering any tail shorter than `check_every_n_chunks`)
        purely for the GUARDRAIL span's record -- by then nothing can be
        blocked, since everything has already been yielded.

    Args:
        feedback: The feedback object to use for blocking. It must only take a single argument.
        threshold: The minimum feedback value required for a context to be included.
        return_value: The value to return (or, for a streaming function,
            yield once in place of further chunks) if the output is
            blocked. Defaults to None.
        check_every_n_chunks: For streaming (generator) functions only: how
            often, in number of chunks yielded by the wrapped function, to
            re-evaluate the feedback against the accumulated output so far.
            Defaults to 20. Ignored for non-streaming functions, which are
            only ever checked once, against the complete output.

    Example:
        ```python
        from trulens.core.guardrails.base import block_output

        feedback = Feedback(provider.criminality, higher_is_better = False)

        class safe_output_chat_app:
            @instrument()
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

        Streaming example:
        ```python
        class safe_streaming_chat_app:
            @instrument(span_type=SpanAttributes.SpanType.GENERATION)
            @block_output(
                feedback=feedback,
                threshold=0.5,
                return_value="[response withheld]",
                check_every_n_chunks=10,
            )
            def chat(self, question: str):
                for chunk in oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    stream=True,
                    messages=[{"role": "user", "content": question}],
                ):
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        ```
    """

    def __init__(
        self,
        feedback: core_metric.Metric,
        threshold: float,
        return_value: Optional[str] = None,
        check_every_n_chunks: int = 20,
    ):
        self.feedback = feedback
        self.threshold = threshold
        self.return_value = return_value
        self.check_every_n_chunks = check_every_n_chunks

    def _evaluate(self, text) -> tuple:
        """Evaluate feedback against `text`; returns (result, blocked)."""
        result = self.feedback(text)
        if not isinstance(result, float):
            raise ValueError(
                "`block_output` can only be used with feedback functions that return a float."
            )
        blocked = (
            self.feedback.higher_is_better and result < self.threshold
        ) or (not self.feedback.higher_is_better and result > self.threshold)
        return result, blocked

    def _check(self, guardrail_name: str, text) -> bool:
        """Run one guardrail check in its own span; returns whether blocked."""
        with _guardrail_span(guardrail_name, self.threshold) as span:
            result, blocked = self._evaluate(text)
            span.set_attribute(SpanAttributes.GUARDRAIL.SCORE, result)
            span.set_attribute(SpanAttributes.GUARDRAIL.PASSED, not blocked)
        return blocked

    def __call__(self, func):
        sig = inspect.signature(func)
        guardrail_name = getattr(self.feedback, "name", func.__name__)

        if inspect.isasyncgenfunction(func):

            async def async_gen_wrapper(*args, **kwargs):
                accumulated = []
                since_last_check = 0
                async for chunk in func(*args, **kwargs):
                    accumulated.append(chunk)
                    since_last_check += 1
                    yield chunk
                    if since_last_check >= self.check_every_n_chunks:
                        since_last_check = 0
                        if self._check(
                            guardrail_name, "".join(map(str, accumulated))
                        ):
                            if self.return_value is not None:
                                yield self.return_value
                            return
                if since_last_check:
                    # Tail shorter than a full checkpoint: still recorded
                    # for observability, but nothing left to block.
                    self._check(guardrail_name, "".join(map(str, accumulated)))

            async_gen_wrapper.__name__ = func.__name__
            async_gen_wrapper.__doc__ = func.__doc__
            async_gen_wrapper.__signature__ = sig
            return async_gen_wrapper

        if inspect.isgeneratorfunction(func):

            def gen_wrapper(*args, **kwargs):
                accumulated = []
                since_last_check = 0
                for chunk in func(*args, **kwargs):
                    accumulated.append(chunk)
                    since_last_check += 1
                    yield chunk
                    if since_last_check >= self.check_every_n_chunks:
                        since_last_check = 0
                        if self._check(
                            guardrail_name, "".join(map(str, accumulated))
                        ):
                            if self.return_value is not None:
                                yield self.return_value
                            return
                if since_last_check:
                    self._check(guardrail_name, "".join(map(str, accumulated)))

            gen_wrapper.__name__ = func.__name__
            gen_wrapper.__doc__ = func.__doc__
            gen_wrapper.__signature__ = sig
            return gen_wrapper

        def wrapper(*args, **kwargs):
            # Run the decorated function first; the guardrail span wraps only
            # the feedback evaluation so its duration reflects the check, not
            # the application logic.
            output = func(*args, **kwargs)
            blocked = self._check(guardrail_name, output)
            return self.return_value if blocked else output

        # note: the following information is manually written to the wrapper because @functools.wraps(func) causes breaking of the method.
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__signature__ = sig
        return wrapper

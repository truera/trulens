import dataclasses
from opentelemetry.semconv.ai import SpanAttributes as ai_span

# TODO: structured data types for attributes?
"""
@dataclasses.dataclass
class RetrieverQuery:
    text: str
    embedding: Optional[List[float]]

@dataclasses.dataclass
class RetrieverContext:
    text: str
    score: Optional[float]
    embedding: Optional[List[float]]
"""

# TODO: semantic conventions from otel

class SpanVectorDBOTEL(SpanTyped):
    """VectorDB attributes from OpenTelemetry Semantic Conventions for AI.

    See [OpenTelemetry Semantic Conventions for AI
    constants](https://github.com/traceloop/openllmetry/blob/main/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv/ai/__init__.py)

    !!! Warning

        The OpenTelemetry Semantic Conventions for AI are still in development.
        Lets not rely on this class until things calm down.
    """

    # span attributes

    vector_db_vendor = Span.attribute_property(ai_span.VECTOR_DB_VENDOR, str) # optionality not known

    vector_db_query_top_k = Span.attribute_property(ai_span.VECTOR_DB_QUERY_TOP_K, int) # optionality not known

    # events

    """
    ai_span.DB_QUERY_EMBEDDINGS
    ai_span.DB_QUERY_RESULT
    """

    # event attributes

    """
    ai_span.DB_QUERY_EMBEDDINGS_VECTOR
    ai_span.DB_QUERY_RESULT_ID
    ai_span.DB_QUERY_RESULT_SCORE
    ai_span.DB_QUERY_RESULT_DISTANCE
    ai_span.DB_QUERY_RESULT_METADATA
    ai_span.DB_QUERY_RESULT_VECTOR
    ai_span.DB_QUERY_RESULT_DOCUMENT
    """


class SpanLLMOTEL(SpanTyped):
    """LLM attributes from OpenTelemetry Semantic Conventions for AI.

    See Open Telemetry Semantic Convetions for AI
    [llm_spans.md](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/llm-spans.md)
    and
    [constants](https://github.com/traceloop/openllmetry/blob/main/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv/ai/__init__.py)

    There is also Arize's [openinference
    conventions](https://github.com/Arize-ai/openinference/blob/main/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py).
    
    !!! Warning

        The OpenTelemetry Semantic Conventions for AI are still in development.
        Lets not rely on this class until things calm down.
    """

    request_model = Span.attribute_property(ai_span.LLM_REQUEST_MODEL, str)

    # system = Span.attribute_property(ai_span.LLM_REQUEST_SYSTEM, str)

    request_max_tokens = Span.attribute_property(ai_span.LLM_REQUEST_MAX_TOKENS, Optional[int])

    request_temperature = Span.attribute_property(ai_span.LLM_REQUEST_TEMPERATURE, Optional[float])

    request_top_p = Span.attribute_property(ai_span.LLM_REQUEST_TOP_P, Optional[float])

    response_finish_reasons = Span.attribute_property("gen_ai.response.finish_reasons", Optional[List[str]])
    # Not yet a constant in otel sem ai

    # response_id = Span.attribute_property("gen_ai.response.id", Optional[str])
    # Not yet a constant in otel sem ai

    response_model = Span.attribute_property(ai_span.LLM_RESPONSE_MODEL, Optional[str])

    # usage_completion_tokens = Span.attribute_property(ai_span.LLM_RESPONSE_COMPLETION_TOKENS, Optional[int])

    # usage_prompt_tokens = Span.attribute_property(ai_span.LLM_RESPONSE_USAGE_PROMPT_TOKENS, Optional[int])

    """
    # These below are in the constants file but are not described in the
    # markdown so their details or optionality is not know.

    request_type = Span.attribute_property(ai_span.LLM_REQUEST_TYPE, Optional[str]) # optionality not known

    usage_total_tokens = Span.attribute_property(ai_span.LLM_USAGE_TOTAL_TOKENS, Optional[int]) # optionality not known

    user = Span.attribute_property(ai_span.LLM_REQUEST_USER, Optional[str]) # optionality not known

    headers = Span.attribute_property(ai_span.LLM_REQUEST_HEADERS, Optional[str]) # optionality, type not known

    top_k = Span.attribute_property(ai_span.LLM_REQUEST_TOP_K, Optional[int]) # optionality not known

    is_streaming = Span.attribute_property(ai_span.LLM_REQUEST_IS_STREAMING, Optional[bool]) # optionality not known

    frequency_penalty = Span.attribute_property(ai_span.LLM_REQUEST_FREQUENCY_PENALTY, Optional[float]) # optionality not known

    presence_penalty = Span.attribute_property(ai_span.LLM_REQUEST_PRESENCE_PENALTY, Optional[float]) # optionality not known

    chat_stop_sequences = Span.attribute_property(ai_span.LLM_REQUEST_CHAT_STOP_SEQUENCES, Optional[List[str]]) # optionality, type not known

    request_functions = Span.attribute_property(ai_span.LLM_REQUEST_FUNCTIONS, Optional[List[str]]) # optionality, type not known
    """

    # Events in description but not yet in python package
    # Events named "gen_ai.content.prompt"
    # Attributes:
    # - "gen_ai.prompt": str (json data)
    # - "gen_ai.completion": str (json data)

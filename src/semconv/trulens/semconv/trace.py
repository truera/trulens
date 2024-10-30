"""Semantic conventions for TruLens data.

This file should not have any dependencies so it can be easily imported by tools
that want to read TruLens data but not use TruLens otherwise.
"""


class SpanAttributes:
    class CALL:
        """Instrumented method call attributes."""

        base = "call"

        CALL_ID = base + ".call_id"
        """Attribute key for call id."""

        STACK = base + ".stack"
        """Attribute key for call stack."""

        SIG = base + ".sig"
        """Attribute key for function signature."""

    class RETRIEVAL:
        """A retrieval."""

        base = "retrieval"

        QUERY_TEXT = base + ".query_text"
        """Input text whose related contexts are being retrieved."""

        QUERY_EMBEDDING = base + ".query_embedding"
        """Embedding of the input text."""

        DISTANCE_TYPE = base + ".distance_type"
        """Distance function used for ranking contexts."""

        NUM_CONTEXTS = base + ".num_contexts"
        """The number of contexts requested, not necessarily retrieved."""

        RETRIEVED_CONTEXTS = base + ".retrieved_contexts"
        """The retrieved contexts."""

        RETRIEVED_SCORES = base + ".retrieved_scores"
        """The scores of the retrieved contexts."""

        RETRIEVED_EMBEDDINGS = base + ".retrieved_embeddings"
        """The embeddings of the retrieved contexts."""

    class RERANKING:
        """A reranker call."""

        base = "reranking"

        QUERY_TEXT = base + ".query_text"
        """The query text."""

        MODEL_NAME = base + ".model_name"
        """The model name of the reranker."""

        TOP_N = base + ".top_n"
        """The number of contexts to rerank."""

        INPUT_CONTEXT_TEXTS = base + ".input_context_texts"
        """The contexts being reranked."""

        INPUT_CONTEXT_SCORES = base + ".input_context_scores"
        """The scores of the input contexts."""

        OUTPUT_RANKS = base + ".output_ranks"
        """Reranked indexes into `input_context_texts`."""

    class GENERATION:
        base = "generation"

        # GEN_AI_*

        MODEL_NAME = base + ".model_name"
        """The model name of the LLM."""
        # GEN_AI_REQUEST_MODEL
        # GEN_AI_RESPONSE_MODEL ?

        MODEL_TYPE = base + ".model_type"
        """The type of model used."""

        INPUT_TOKEN_COUNT = base + ".input_token_count"
        """The number of tokens in the input."""
        # GEN_AI_USAGE_INPUT_TOKENS

        INPUT_MESSAGES = base + ".input_messages"
        """The prompt given to the LLM."""
        # GEN_AI_PROMPT

        OUTPUT_TOKEN_COUNT = base + ".output_token_count"
        """The number of tokens in the output."""
        # GEN_AI_USAGE_OUTPUT_TOKENS

        OUTPUT_MESSAGES = base + ".output_messages"
        """The returned text."""

        TEMPERATURE = base + ".temperature"
        """The temperature used for generation."""
        # GEN_AI_REQUEST_TEMPERATURE

        COST = base + ".cost"
        """The cost of the generation."""

    class MEMORIZATION:
        """A memory saving call."""

        base = "memorization"

        MEMORY_TYPE = base + ".memory_type"
        """The type of memory."""

        REMEMBERED = base + ".remembered"
        """The text being integrated into the memory in this span."""

    class EMBEDDING:
        """An embedding call."""

        base = "embedding"

        INPUT_TEXT = base + ".input_text"
        """The text being embedded."""

        MODEL_NAME = base + ".model_name"
        """The model name of the embedding model."""

        EMBEDDING = base + ".embedding"
        """The embedding of the input text."""

    class TOOL_INVOCATION:
        """A tool invocation."""

        base = "tool_invocation"

        DESCRIPTION = base + ".description"
        """The description of the tool."""

    class AGENT_INVOCATION:
        """An agent invocation."""

        base = "agent_invocation"

        DESCRIPTION = base + ".description"
        """The description of the agent."""

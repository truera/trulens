from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from typing_extensions import TypedDict

from .config import Settings
from .config import get_settings
from .embeddings import OpenAIEmbeddings
from .embeddings import _get_client


class RAGState(TypedDict, total=False):
    question: str
    context: list[Document]
    answer: str


@lru_cache(maxsize=None)
def _vector_store_cached(
    embedding_model: str,
    collection_name: str,
    qdrant_url: str,
    qdrant_api_key: str,
    openai_api_key: str,
) -> QdrantVectorStore:
    """Reuse a single Qdrant connection per (model, collection, url) tuple.

    from_existing_collection opens a remote connection and does a collection
    round trip; caching avoids repeating that on every retrieval call.
    """
    return QdrantVectorStore.from_existing_collection(
        embedding=OpenAIEmbeddings(embedding_model, openai_api_key),
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
    )


def _vector_store(settings: Settings) -> QdrantVectorStore:
    return _vector_store_cached(
        settings.embedding_model,
        settings.collection_name,
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.openai_api_key,
    )


@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: (
            kwargs.get("question") or (args[0] if args else "")
        ),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: (
            [doc.page_content for doc in ret] if ret else []
        ),
    },
)
def retrieve_context(
    question: str, settings: Settings | None = None
) -> list[Document]:
    settings = settings or get_settings()
    return _vector_store(settings).similarity_search(
        question, k=settings.retrieval_k
    )


@instrument(span_type=SpanAttributes.SpanType.GENERATION)
def generate_answer(
    question: str, context: list[Document], settings: Settings | None = None
) -> str:
    settings = settings or get_settings()
    formatted_context = "\n\n".join(
        f"[page {document.metadata.get('page', '?') + 1}] {document.page_content}"
        for document in context
    )
    system_prompt = (
        "You are an aircraft systems reference assistant. Answer only from the "
        "supplied context. If the context does not contain the answer, say you do "
        "not know rather than guessing at a procedure, limitation, or value."
        f"Context:\n{formatted_context}"
    )
    client = _get_client(settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ],
    )
    return response.choices[0].message.content or ""


def build_graph(settings: Settings | None = None):
    settings = settings or get_settings()

    def retrieve_node(state: RAGState) -> RAGState:
        return {"context": retrieve_context(state["question"], settings)}

    def answer_node(state: RAGState) -> RAGState:
        return {
            "answer": generate_answer(
                state["question"], state["context"], settings
            )
        }

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def answer_question(question: str, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    result = build_graph(settings).invoke({"question": question})
    return {
        "question": question,
        "answer": result["answer"],
        "context": [document.page_content for document in result["context"]],
    }

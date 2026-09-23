from langchain_community.document_loaders import PyPDFium2Loader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .config import get_settings
from .embeddings import get_embeddings


def index_pdf(settings: Settings | None = None) -> int:
    """Index the source PDF into Qdrant Cloud."""
    settings = settings or get_settings()
    if not settings.source_pdf.exists():
        raise FileNotFoundError(
            f"Source PDF not found at {settings.source_pdf}. Attach the document there."
        )

    documents = PyPDFium2Loader(str(settings.source_pdf)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    QdrantVectorStore.from_documents(
        chunks,
        get_embeddings(settings),
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        force_recreate=True,
    )
    return len(chunks)

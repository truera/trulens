from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    source_pdf: Path = Path("data/aircraft_systems.pdf")
    collection_name: str = "aircraft_systems"
    embedding_model: str = "jinaai/jina-embeddings-v2-small-en"
    chat_model: str = "gemini-3.5-flash"
    eval_model: str = "gpt-5.6-luna"
    chunk_size: int = 1024
    chunk_overlap: int = 150
    retrieval_k: int = 4

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    database_url: str | None = os.getenv("TRULENS_DATABASE_URL", None)


def get_settings() -> Settings:
    return Settings()

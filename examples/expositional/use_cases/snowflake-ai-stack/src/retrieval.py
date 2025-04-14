import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, embed_model_name=None, embed_dimension=None):
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.embed_model_name = embed_model_name or "Snowflake/snowflake-arctic-embed-l-v2.0"
        self.embed_dimension = embed_dimension or 512  # Adjust as necessary
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_name)
        self.client = chromadb.Client(Settings(persist_directory="./chroma_vector_store"))
        self.collection = self.client.get_or_create_collection(name="snowflake_eng_blogs")

    def load_text_files(self, file_path):
        """Load text files from a list of paths using TextLoader."""
        loader = TextLoader(file_path=file_path)
        docs = loader.load()
        return docs

    def load_documents(self, urls):
        """Load documents from a list of URLs using WebBaseLoader."""
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return docs

    def load_persisted(self):
        """Return the number of vectors loaded from the persisted chromadb vector store."""
        data = self.collection.get()
        return len(data["ids"])


    def split_documents(self, documents):
        """Split the loaded documents into chunks using SemanticChunker."""
        text_splitter = SemanticChunker(OpenAIEmbeddings())
        chunks = text_splitter.split_documents(documents=documents)
        return chunks

    def add_chunks(self, chunks, batch_size=16):
        """Filter chunks and add them to the vector store."""
        filtered_chunks = filter_complex_metadata(chunks)
        documents = []
        metadatas = []
        ids = []
        for chunk in filtered_chunks:
            # Assuming each chunk has .page_content and .metadata attributes
            documents.append(chunk.page_content)
            metadatas.append(chunk.metadata)
            ids.append(str(uuid.uuid4()))

        embeddings = []
        # Process documents in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch_texts = documents[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

    def search(self, query, k=5):
        """Perform a similarity search on the vector store using the query and return LangChain documents."""
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        # Create documents and deduplicate based on page_content
        unique_docs = {}
        for doc_text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            if doc_text not in unique_docs:
                unique_docs[doc_text] = Document(page_content=doc_text, metadata=metadata)

        docs = list(unique_docs.values())
        return docs

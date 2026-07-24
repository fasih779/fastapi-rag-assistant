import os
import time
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
from langchain_qdrant import QdrantVectorStore
from chunker import RAGProcessor

load_dotenv()

class EmbeddingModel:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding = HuggingFaceEmbeddings(model_name=model)

    def get_model(self):
        return self.embedding

class QdrantStore:
    def __init__(
        self,
        embedding_model,
        collection_name="fastapi_codebase"
    ):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Load credentials
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY") or os.getenv("qudrant")
        
        if self.url:
            print(f"Connecting to Qdrant Cloud at {self.url}...")
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            print("No QDRANT_URL found in .env. Falling back to local Qdrant storage (local_qdrant)...")
            self.client = QdrantClient(path="local_qdrant")

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100
            )
        )

    def add_documents(self, documents):
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        vector_store.add_documents(documents)
        return vector_store

    def get_retriever(self, k=8, search_type="similarity"):
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

if __name__ == "__main__":
    path = r"D:\RAG Project\fastapi"
    processor = RAGProcessor(path)
    documents = processor.load_documents()
    chunks = processor.create_chunks(documents)
    
    embedder = EmbeddingModel()
    store = QdrantStore(embedding_model=embedder.get_model())
    
    # Process chunks in batches of 500 (locally processed, no rate limits!)
    batch_size = 500
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"Embedding {len(chunks)} chunks in {total_batches} batches...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        store.add_documents(batch)
        print(f"Added batch {i//batch_size + 1}/{total_batches}")
    
    retriever = store.get_retriever(k=3)
    results = retriever.invoke("FastAPI")
    if results:
        print("\nTest Retrieval Result:")
        print(results[0].page_content[:150])
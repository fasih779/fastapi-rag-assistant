import os
import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag_chain import RAGChain

app = FastAPI(
    title="RAG API",
    description="Query any indexed codebase using RAG + Llama 3.3-70B",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache: one RAGChain per collection so we don't re-init on every request
_rag_cache: dict[str, RAGChain] = {}
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "fastapi_codebase")


def get_rag(collection: str) -> RAGChain:
    if collection not in _rag_cache:
        _rag_cache[collection] = RAGChain(collection_name=collection)
    return _rag_cache[collection]


class QueryRequest(BaseModel):
    question: str
    collection: Optional[str] = None   # defaults to DEFAULT_COLLECTION


class QueryResponse(BaseModel):
    answer: str
    collection: str


@app.on_event("startup")
def startup():
    # Pre-warm the default collection
    get_rag(DEFAULT_COLLECTION)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    collection = request.collection or DEFAULT_COLLECTION
    rag = get_rag(collection)
    answer = rag.query(request.question)
    return QueryResponse(answer=answer, collection=collection)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_collections": list(_rag_cache.keys())}

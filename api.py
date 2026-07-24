from networkx.algorithms.flow import preflowpush
import os
import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import RAGChain

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.on_event("startup")
def startup():
    global rag
    rag = RAGChain()

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    answer = rag.query(request.question)
    return QueryResponse(answer=answer)

@app.get("/health")
def health():
    return {"status": "ok"}

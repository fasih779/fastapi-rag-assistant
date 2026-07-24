import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from embedding import EmbeddingModel, QdrantStore
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RAGChain:
    def __init__(self, model_name="llama-3.3-70b-versatile", collection_name="fastapi_codebase"):
        self.embedder = EmbeddingModel()
        self.store = QdrantStore(
            embedding_model=self.embedder.get_model(),
            collection_name=collection_name
        )
        self.retriever = self.store.get_retriever(k=8, search_type="similarity")
        
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. You MUST ALWAYS write your entire response in English. Even if the retrieved context is in Korean, Chinese, or any other language, read and translate the information into a clear English answer."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        return self.chain.invoke(question)

if __name__ == "__main__":
    history = []
    rag = RAGChain()
    while True:
        query=str(input("Ask a question: "))
        
        if query.lower() == "exit":
            break
        response = rag.query(query)
        print(response)
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=response))

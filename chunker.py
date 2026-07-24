from pathlib import Path
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter
)

class RAGProcessor:
    def __init__(self, path: str):
        self.path = Path(path)
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1500,
            chunk_overlap=200
        )
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3")
            ]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

    def load_documents(self):
        documents = []
        extensions = [".py", ".md", ".rst"]
        for file in self.path.rglob("*"):
            if file.is_file() and file.suffix in extensions:
                try:
                    content = file.read_text(encoding="utf-8")
                    documents.append({
                        "path": str(file),
                        "extension": file.suffix,
                        "type": "code" if file.suffix == ".py" else "documentation",
                        "content": content
                    })
                except Exception as e:
                    pass
        return documents

    def create_chunks(self, documents):
        all_chunks = []
        for document in documents:
            if document["type"] == "code":
                split_docs = self.code_splitter.create_documents(
                    texts=[document["content"]],
                    metadatas=[{
                        "source": document["path"],
                        "type": "code",
                        "extension": document["extension"]
                    }]
                )
                all_chunks.extend(split_docs)
            else:
                md_docs = self.md_splitter.split_text(document["content"])
                if not md_docs:
                    md_docs = self.text_splitter.create_documents([document["content"]])
                split_docs = self.text_splitter.split_documents(md_docs)
                for chunk in split_docs:
                    chunk.metadata.update({
                        "source": document["path"],
                        "type": "documentation",
                        "extension": document["extension"]
                    })
                all_chunks.extend(split_docs)
        return all_chunks

if __name__ == "__main__":
    path = r"D:\RAG Project\fastapi"
    processor = RAGProcessor(path)
    documents = processor.load_documents()
    chunks = processor.create_chunks(documents)
    print("Total chunks:", len(chunks))
    if chunks:
        print(chunks[0].page_content[:100])
        print(chunks[0].metadata)
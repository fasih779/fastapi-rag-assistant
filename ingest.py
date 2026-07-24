"""
ingest.py — Index ANY codebase into Qdrant.

Usage:
    python ingest.py --path ./my_project --collection my_project
    python ingest.py --path D:/repos/django --collection django_docs
    python ingest.py --path ./fastapi --collection fastapi_codebase

Supported file types: .py  .md  .rst
"""

import argparse
from embedding import EmbeddingModel, QdrantStore
from chunker import RAGProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Index a codebase into Qdrant for RAG querying."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the folder you want to index (e.g. ./my_project)"
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Name for the Qdrant collection (e.g. my_project)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of chunks to embed per batch (default: 500)"
    )
    args = parser.parse_args()

    print(f"\n📂 Indexing path   : {args.path}")
    print(f"📦 Collection name : {args.collection}")
    print(f"⚡ Batch size      : {args.batch_size}\n")

    # Step 1 — Load & chunk the codebase
    print("🔍 Loading and chunking documents...")
    processor = RAGProcessor(args.path)
    documents = processor.load_documents()

    if not documents:
        print("❌ No .py / .md / .rst files found at that path. Check the folder and try again.")
        return

    chunks = processor.create_chunks(documents)
    print(f"✅ {len(documents)} files → {len(chunks)} chunks\n")

    # Step 2 — Embed & store in Qdrant
    print("🧠 Embedding and storing in Qdrant...")
    embedder = EmbeddingModel()
    store = QdrantStore(
        embedding_model=embedder.get_model(),
        collection_name=args.collection
    )

    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i : i + args.batch_size]
        store.add_documents(batch)
        batch_num = i // args.batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} done")

    print(f"\n🎉 Done! Collection '{args.collection}' is ready to query.")
    print(f"   Start the API:  uvicorn api:app --port 8000")
    print(f"   Then ask:       POST /query  {{\"question\": \"...\", \"collection\": \"{args.collection}\"}}\n")


if __name__ == "__main__":
    main()

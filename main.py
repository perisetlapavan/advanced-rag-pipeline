"""
main.py — Advanced Enterprise RAG Pipeline (Groq + LangChain + BGE)

Runs the pipeline standalone with a handful of demo queries — useful for a
quick sanity check outside the eval harness. For evaluation, use
`python src/runner.py` instead.

Usage:
    python main.py
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document

from pipeline.ingestor import TechnicalDocIngestor
from pipeline.retriever import AdvancedTechnicalRetriever
from pipeline.engine import TechnicalRAGPipeline

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY not found. Add it to your .env file before running."
    )


def load_documents_from_data_dir(data_dir: str = "./data") -> list[Document]:
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filename}))
            print(f"  [Loader] Loaded: {filename} ({len(content)} chars)")
    return docs


def main():
    print("=" * 60)
    print("🚀 Advanced Enterprise RAG Pipeline  |  Powered by Groq")
    print("=" * 60)

    raw_documents = load_documents_from_data_dir("./data")

    print("\n📦 Step 1 — Chunking & indexing documents...")
    ingestor = TechnicalDocIngestor(persist_directory="./chroma_db")
    vector_db = ingestor.process_documents(raw_documents)

    print("\n🔍 Step 2 — Mounting hybrid BM25 + Vector + Reranker retriever...")
    retriever = AdvancedTechnicalRetriever(vector_db, raw_documents)

    print("\n🧠 Step 3 — Initialising RAG orchestration engine...")
    pipeline = TechnicalRAGPipeline(retriever)

    queries = [
        "What is the register for system status and what port does it use?",
        "What happens if pressure runs at 500 kPa for 10 seconds?",
        "What is the maximum torque limit for the generator shaft?",  # Not in context
    ]

    print("\n" + "=" * 60)
    print("📋 Running Validation Queries")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 50)
        output = pipeline.ask(query)
        print(output)
        print("-" * 50)

    print("\n✅ Pipeline run complete.")


if __name__ == "__main__":
    main()

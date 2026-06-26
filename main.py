"""
main.py — Advanced Enterprise RAG Pipeline (Groq + LangChain + BGE)

Usage:
    python main.py

To use your own documents, replace the mock_spec string (or load .txt / .pdf
files from the /data folder) and pass them as Document objects to
ingestor.process_documents().
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document

from pipeline.ingestor import TechnicalDocIngestor
from pipeline.retriever import AdvancedTechnicalRetriever
from pipeline.engine import TechnicalRAGPipeline

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY not found. Add it to your .env file before running."
    )


def load_documents_from_data_dir(data_dir: str = "./data") -> list[Document]:
    """Load all .txt files from the /data directory as Document objects."""
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

    # ── 1. Load Documents ──────────────────────────────────────────────────────
    # Try loading from /data first; fall back to inline mock spec for demo runs.
    raw_documents = load_documents_from_data_dir("./data")

    if not raw_documents:
        print("\n  [Loader] No .txt files in /data — using inline mock specification.")
        mock_spec = """
        SPECIFICATION ID: SPEC-2026-X9
        SYSTEM: High-Pressure Cooling Assembly

        OPERATIONAL TEMPERATURE LIMIT: Max 180°C, Min -10°C.

        CRITICAL SHUTDOWN CRITERIA:
        If pressure exceeds 450 kPa for more than 4 seconds, trip valve V-102
        must isolate the coolant loop immediately to prevent thermal runaway.

        COMMUNICATION PROTOCOL:
        Modbus TCP over Port 502.
        Register 40001 stores the system status word.
        Register 40002 stores current pressure in kPa (unsigned 16-bit integer).

        MAINTENANCE INTERVAL: Full inspection every 2000 operating hours.
        COOLANT TYPE: Deionized water with corrosion inhibitor Grade CI-7.
        """
        raw_documents = [
            Document(page_content=mock_spec, metadata={"source": "mock_spec_SPEC-2026-X9.txt"})
        ]

    # ── 2. Ingest + Build Vector DB ────────────────────────────────────────────
    print("\n📦 Step 1 — Chunking & indexing documents...")
    ingestor = TechnicalDocIngestor(persist_directory="./chroma_db")
    vector_db = ingestor.process_documents(raw_documents)

    # ── 3. Mount Hybrid Retriever ──────────────────────────────────────────────
    print("\n🔍 Step 2 — Mounting hybrid BM25 + Vector + Reranker retriever...")
    retriever = AdvancedTechnicalRetriever(vector_db, raw_documents)

    # ── 4. Init RAG Pipeline ───────────────────────────────────────────────────
    print("\n🧠 Step 3 — Initialising RAG orchestration engine...")
    pipeline = TechnicalRAGPipeline(retriever)

    # ── 5. Run Queries ─────────────────────────────────────────────────────────
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

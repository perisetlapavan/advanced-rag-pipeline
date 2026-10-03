"""
The system under test — now wired to the real RAG pipeline instead of a stub.

Builds the pipeline (ingest -> hybrid retriever -> Groq engine) exactly once
per process and reuses it across every test case, since ingestion + loading
the embedding/reranker models is the expensive part.

Two things this adapter does that main.py doesn't:
  1. Calls retriever.retrieve() separately so the eval harness can score
     groundedness against what was ACTUALLY retrieved for that question,
     not a hardcoded context string.
  2. Parses the raw chain-of-thought response down to just the FINAL ANSWER
     section, so the judge and guardrails score the answer itself rather
     than the THINKING/VERIFICATION scratch work.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from pipeline.ingestor import TechnicalDocIngestor
from pipeline.retriever import AdvancedTechnicalRetriever
from pipeline.engine import TechnicalRAGPipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

_pipeline = None
_retriever = None


def _load_documents() -> list[Document]:
    docs = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filename}))
            print(f"  [Loader] Loaded: {filename} ({len(content)} chars)")
    return docs


def _build_pipeline():
    """Ingest + index + mount the pipeline once. Cached at module level."""
    global _pipeline, _retriever
    if _pipeline is not None:
        return _pipeline, _retriever

    docs = _load_documents()
    if not docs:
        raise RuntimeError(
            f"No .txt files found in {DATA_DIR}. Add your technical documents "
            "there (or keep the provided example_standard.txt for a first run)."
        )

    print("\n📦 Ingesting documents...")
    ingestor = TechnicalDocIngestor(persist_directory=CHROMA_DIR)
    vector_db = ingestor.process_documents(docs)

    print("\n🔍 Mounting hybrid retriever...")
    _retriever = AdvancedTechnicalRetriever(vector_db, docs)

    print("\n🧠 Initialising RAG pipeline...")
    _pipeline = TechnicalRAGPipeline(_retriever)

    return _pipeline, _retriever


def _extract_final_answer(raw_response: str) -> str:
    """Pull just the FINAL ANSWER section out of the chain-of-thought response."""
    if "INSUFFICIENT_LOCAL_CONTEXT" in raw_response:
        return "INSUFFICIENT_LOCAL_CONTEXT"
    marker = "FINAL ANSWER:"
    if marker in raw_response:
        return raw_response.split(marker, 1)[1].strip()
    # Fallback: no marker found, return the raw response as-is
    return raw_response.strip()


def generate_answer(question: str, top_n: int = 3) -> dict:
    """Run the real pipeline end to end and return everything the eval
    harness needs: the parsed final answer, the raw response (kept for
    debugging), and the context the retriever actually pulled.
    """
    pipeline, retriever = _build_pipeline()

    retrieved_chunks = retriever.retrieve(question, top_n=top_n)
    context = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else ""

    raw_response = pipeline.ask(question, top_n=top_n)
    final_answer = _extract_final_answer(raw_response)

    return {
        "answer": final_answer,
        "raw_response": raw_response,
        "context": context,
        "n_retrieved": len(retrieved_chunks),
    }

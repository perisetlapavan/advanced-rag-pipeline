# Advanced Enterprise RAG Pipeline
**Groq (LLaMA 3.3 70B) + LangChain + BGE Embeddings + Cross-Encoder Reranker**

A production-grade Retrieval-Augmented Generation pipeline targeting technical document Q&A with grounded, non-hallucinated answers.

---

## Architecture

```
Documents (.txt / .pdf)
        │
        ▼
  TechnicalDocIngestor          ← Hierarchical parent-child chunking
  (BAAI/bge-large-en-v1.5)     ← Local embeddings (no OpenAI dependency)
        │
        ▼
   ChromaDB (persisted)
        │
        ▼
AdvancedTechnicalRetriever
  ├── Dense Vector Search  (k=8)
  ├── Sparse BM25 Search   (k=8)
  └── CrossEncoder Reranker  ← BAAI/bge-reranker-large
        │
        ▼
  TechnicalRAGPipeline
  (Groq — llama-3.3-70b-versatile, temp=0.0)
  Chain-of-Thought prompt: THINK → VERIFY → ANSWER
        │
        ▼
  Grounded Technical Answer
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads ~1.5 GB of model weights for the BGE embedding model
> and cross-encoder reranker. These are cached locally after the first download.

### 2. Configure environment

Edit `.env`:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Add your documents

Drop `.txt` files into the `/data` folder. The pipeline loads all of them automatically.

### 4. Run

```bash
python main.py
```

---

## Project Structure

```
advanced_rag_pipeline/
├── data/                        # Drop .txt technical documents here
│   └── example_standard.txt
├── pipeline/
│   ├── __init__.py
│   ├── ingestor.py              # Hierarchical child-parent chunking → Chroma
│   ├── retriever.py             # BM25 + Vector Search + Cross-Encoder reranker
│   └── engine.py                # Groq LLM + Chain-of-Thought RAG loop
├── main.py                      # Entry point
├── requirements.txt
├── .env                         # GROQ_API_KEY goes here
└── README.md
```

---

## Key Design Decisions

| Choice | Rationale |
|---|---|
| `BAAI/bge-large-en-v1.5` embeddings | State-of-the-art open-source, runs locally, no API cost |
| Hierarchical parent-child chunking | Child chunks give precision; parent context gives the LLM full surrounding detail |
| BM25 + Vector hybrid | BM25 catches exact keyword hits (register numbers, spec IDs); dense search catches semantic paraphrases |
| `BAAI/bge-reranker-large` | Cross-encoder reranker significantly improves precision over raw retrieval scores |
| `temperature=0.0` | Technical specs demand deterministic, reproducible outputs |
| `INSUFFICIENT_LOCAL_CONTEXT` guard | Forces the LLM to admit when context is absent rather than hallucinate |

---

## Extending the Pipeline

- **Load PDFs**: use `langchain_community.document_loaders.PyPDFLoader` and pass results to `ingestor.process_documents()`
- **Persist across runs**: Chroma is already persisted to `./chroma_db`. Skip re-ingestion by loading the existing DB with `Chroma(persist_directory="./chroma_db", embedding_function=...)`
- **Add more queries**: extend the `queries` list in `main.py` or wrap `pipeline.ask()` in a FastAPI endpoint

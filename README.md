# Enterprise RAG Pipeline — Evaluated & Guardrailed

**Groq (openai/gpt-oss-120B) + LangChain + BGE Embeddings + Cross-Encoder Reranker**,
wrapped in an evaluation and guardrails harness.

A production-grade RAG pipeline for technical document Q&A, plus the piece
most GenAI portfolios skip: proof it's accurate and proof it can't be
tricked. The pipeline retrieves and answers; the harness in `src/` measures
how well it does both, before and after a guardrail layer is added.

---

## Architecture

```
Documents (.txt)
        │
        ▼
  TechnicalDocIngestor          ← Hierarchical parent-child chunking
  (BAAI/bge-large-en-v1.5)     ← Local embeddings
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
  (Groq — openai/gpt-oss-120B, temp=0.0)
  Chain-of-Thought: THINK → VERIFY → ANSWER
        │
        ▼
  Grounded Answer  ────────────────┐
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │   Eval & Guardrails      │
                      │  ┌─────────────────────┐ │
                      │  │ Input guardrails    │ │  PII regex, prompt-injection
                      │  │ (before the LLM)    │ │  phrase detection
                      │  └─────────────────────┘ │
                      │  ┌─────────────────────┐ │
                      │  │ Output guardrails   │ │  hallucination pre-filter,
                      │  │ (after the LLM)     │ │  refusal-compliance,
                      │  │                     │ │  native-guard adherence
                      │  └─────────────────────┘ │
                      │  ┌─────────────────────┐ │
                      │  │ LLM-as-judge        │ │  correctness + groundedness
                      │  │ scoring             │ │  (1-5, against real context)
                      │  └─────────────────────┘ │
                      └──────────────────────────┘
                                   │
                                   ▼
                    results/results.json → Streamlit dashboard
```

## Why this project

Most GenAI portfolios stop at "I built a RAG chatbot." This one answers the
next two questions an interviewer asks: **how do you know it's accurate**,
and **how do you stop it from doing something you didn't intend?**

The pipeline already has a first line of defense — the chain-of-thought
prompt forces an `INSUFFICIENT_LOCAL_CONTEXT` response instead of
hallucinating when the answer isn't in the retrieved chunks. The eval
harness measures how well that native guard actually performs, and adds a
layer the prompt alone doesn't cover: PII leakage and prompt-injection
resistance, checked *before* the question even reaches the LLM.

## What the harness does

1. **Runs the real pipeline** — `src/rag_system.py` builds the actual
   ingestor → retriever → engine stack once, then evaluates every test case
   against live retrieval (not hardcoded context).
2. **LLM-as-judge evaluation** — scores each answer on correctness and
   groundedness (1-5) against the context the retriever *actually pulled*,
   with a written justification.
3. **Input guardrails** — regex-based PII detection and prompt-injection
   phrase detection, run before the pipeline is called at all.
4. **Output guardrails** — a numeric-claim hallucination pre-filter, a
   refusal-compliance check for adversarial cases, and a check on whether
   the pipeline's *own* `INSUFFICIENT_LOCAL_CONTEXT` guard fired when it
   should have.
5. **Before/after comparison** — every case runs twice (guardrails off, then
   on), so the summary reports a real delta: hallucination rate, adversarial
   block rate, native context-guard rate, and average judge score.
6. **Streamlit dashboard** — per-case results, category breakdown, and the
   headline before/after numbers.

## Results

*(Fill in after running `src/runner.py` — replace this table with your
actual numbers before it goes on your resume.)*

| Metric | Guardrails OFF | Guardrails ON |
|---|---|---|
| Hallucination rate | — | — |
| Adversarial block rate | — | — |
| Native context-guard rate | — | — |
| Avg. judge correctness | — | — |

## Project structure

```
genai-rag-eval/
├── pipeline/                    # The RAG system itself
│   ├── ingestor.py               # Hierarchical child-parent chunking → Chroma
│   ├── retriever.py              # BM25 + Vector + Cross-Encoder reranker
│   └── engine.py                 # Groq LLM + Chain-of-Thought RAG loop
├── data/
│   ├── example_standard.txt      # Sample technical spec (swap for your own .txt docs)
│   └── test_cases.json           # 12 cases across 4 categories
├── src/                          # The eval + guardrails harness
│   ├── rag_system.py              # Adapter: builds the pipeline once, runs it per question
│   ├── guardrails.py               # Input + output guardrail checks
│   ├── eval_metrics.py            # LLM-as-judge + lexical metrics
│   └── runner.py                  # Runs the full suite, writes results/results.json
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── main.py                       # Standalone demo runner (original 3-query sanity check)
├── requirements.txt
├── .env.example
└── results/
    └── results.json              # generated by runner.py
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your GROQ_API_KEY
```

> First run downloads ~1.5 GB of model weights for the embedding model and
> reranker. Cached locally after that.

Drop your own `.txt` documents into `/data` to replace the example spec —
the ingestor loads every `.txt` file in that folder automatically.

## Running it

```bash
# Quick sanity check — 3 demo queries, no eval/guardrails
python main.py

# Full evaluation + guardrails suite
python src/runner.py

# View results in the browser
streamlit run dashboard/app.py
```

## Key design decisions

| Choice | Rationale |
|---|---|
| `BAAI/bge-large-en-v1.5` embeddings | State-of-the-art open-source, runs locally, no API cost |
| Hierarchical parent-child chunking | Child chunks give precision; parent context gives the LLM full surrounding detail |
| BM25 + Vector hybrid | BM25 catches exact keyword hits (register numbers, spec IDs); dense search catches paraphrases |
| `BAAI/bge-reranker-large` | Cross-encoder reranker improves precision over raw retrieval scores |
| `temperature=0.0` | Deterministic, reproducible outputs |
| `INSUFFICIENT_LOCAL_CONTEXT` guard | Forces the LLM to admit when context is absent — the harness measures how often this actually works |
| Pipeline built once, reused per case | Avoids re-running embedding/reranker model loads for every test question |
| Guardrails run on the parsed FINAL ANSWER | The chain-of-thought scratch work (THINKING/VERIFICATION) isn't what a downstream system would consume — only the final answer is checked/scored |

## Extending it

- **Grow the test set**: 12 seed cases across 4 categories is a starting
  point — 30-40 well-labeled cases makes the reported rates more reliable.
- **Load PDFs**: use `langchain_community.document_loaders.PyPDFLoader` and
  pass results into `ingestor.process_documents()`.
- **Add RAGAS**: swap the hand-rolled lexical metrics in `eval_metrics.py`
  for `ragas.metrics` if you want the industry-standard metric names.
- **Add a PII/NER model**: swap the regex PII detector in `guardrails.py`
  for `presidio` for named-entity-based detection.

## Resume framing

> Built and evaluated a production-style RAG pipeline (hybrid BM25 + dense
> retrieval, cross-encoder reranking, Groq/LLaMA) with an LLM-as-judge
> evaluation and guardrails layer — measured the pipeline's native
> hallucination-guard accuracy, added input/output guardrails for prompt
> injection and PII leakage, reducing hallucination rate by X% and blocking
> Y% of adversarial inputs.

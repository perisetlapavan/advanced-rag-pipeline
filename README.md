# Enterprise RAG Pipeline — Evaluated & Guardrailed

**Groq (openai/gpt-oss-120b) + LangChain + BGE Embeddings + Cross-Encoder Reranker**,
wrapped in an evaluation and guardrails harness.

A RAG pipeline for technical document Q&A, plus the piece most GenAI
portfolios skip: a measurement of how accurate it is and how well it resists
being tricked. The pipeline retrieves and answers; the harness in `src/`
measures how well it does both, with and without a guardrail layer.

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
  └── CrossEncoder Reranker  ← BAAI/bge-reranker-large (top 3 parent contexts)
        │
        ▼
  TechnicalRAGPipeline
  (Groq — openai/gpt-oss-120b, temp=0.0)
  Chain-of-Thought: THINKING → VERIFICATION → FINAL ANSWER
        │
        ▼
  Grounded Answer  ────────────────┐
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │   Eval & Guardrails      │
                      │  ┌─────────────────────┐ │
                      │  │ Input guardrails    │ │  PII regex (account no., PAN,
                      │  │ (before the LLM)    │ │  email, phone), injection
                      │  │                     │ │  phrase list
                      │  └─────────────────────┘ │
                      │  ┌─────────────────────┐ │
                      │  │ Output guardrails   │ │  numeric-claim hallucination
                      │  │ (after the LLM)     │ │  pre-filter, refusal-compliance,
                      │  │                     │ │  native-guard adherence
                      │  └─────────────────────┘ │
                      │  ┌─────────────────────┐ │
                      │  │ LLM-as-judge        │ │  correctness + groundedness
                      │  │ (Groq, separate     │ │  (1-5, against the context the
                      │  │  judge model)       │ │  retriever actually pulled)
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
answering beyond the retrieved context. The eval harness measures how well
that native guard performs, and adds a layer the prompt alone doesn't cover:
PII and prompt-injection checks run *before* the question reaches the LLM.

## What the harness does

1. **Runs the real pipeline** — `src/rag_system.py` builds the actual
   ingestor → retriever → engine stack once, then evaluates every test case
   against live retrieval (not hardcoded context).
2. **LLM-as-judge evaluation** — scores each answer on correctness and
   groundedness (1-5) against the context the retriever *actually pulled*,
   with a one-sentence justification. The judge is a separate Groq model,
   defaulting to `llama-3.3-70b-versatile`, configurable via the
   `GROQ_JUDGE_MODEL` environment variable.
3. **Input guardrails** — regex-based PII detection (account-style numbers,
   PAN, email, Indian mobile numbers) and a prompt-injection phrase list,
   run before the pipeline is called at all.
4. **Output guardrails** — a numeric-claim hallucination pre-filter, a
   refusal-compliance check for adversarial cases, and a check on whether
   the pipeline's *own* `INSUFFICIENT_LOCAL_CONTEXT` guard fired when it
   should have.
5. **Before/after comparison** — every case runs twice (guardrails off, then
   on), so the summary reports a delta for hallucination rate, adversarial
   block rate, native context-guard rate, and average judge correctness.
6. **Streamlit dashboard** — headline before/after metrics (hallucination
   rate, adversarial block rate, judge correctness), a per-case table with an
   on/off toggle, and a per-category breakdown. The native context-guard rate
   is in the dashboard's raw summary JSON expander.

## Test set

`data/test_cases.json` is a **12-case suite across 4 categories**, written
against the sample spec in `data/example_standard.txt`:

| Category | Count | IDs | Expected answer |
|---|---|---|---|
| `normal` | 5 | `norm_001`–`norm_005` | The ground-truth answer from the spec |
| `no_answer_in_context` | 2 | `edge_001`–`edge_002` | `INSUFFICIENT_LOCAL_CONTEXT` |
| `prompt_injection` | 3 | `adv_001`–`adv_003` | `REFUSE` |
| `pii_bait` | 2 | `pii_001`–`pii_002` | `REDACT_AND_REFUSE` |

A larger 50-case set (20 normal / 10 out-of-context / 10 injection / 10 PII)
is available in `data/test_cases_extended.json`; see "Extending it".

## Results

Latest run: the 12-case suite, guardrails off vs. on.

| Metric | Guardrails OFF | Guardrails ON | Status |
|---|---|---|---|
| Native context-guard rate (out-of-context queries declined) | 100% (2/2) | 100% (2/2) | Valid |
| Adversarial block rate (injection + PII) | 0% (0/5) | 100% (5/5) | Valid (see note 1) |
| Hallucination rate | 100% | 58.3% | **Invalid** (see note 2) |
| Avg. judge correctness | 0.00 / 5 | 0.00 / 5 | **Invalid** (see note 2) |

1. In the OFF pass an answer only counts as a block when it is the literal
   string `REFUSE`. The model itself declined or returned
   `INSUFFICIENT_LOCAL_CONTEXT` on all 5 adversarial cases in that pass, so
   0% understates the unguarded pipeline. With guardrails on, the input
   checks blocked all 5 before they reached the LLM.
2. Every judge call in that run failed with a `404 model_not_found` for
   `llama-3.3-70b-versatile`, so all judge scores were recorded as 0 and
   every judged case counted as ungrounded. The hallucination rate and
   correctness score reflect the failed judge, not the pipeline. Set
   `GROQ_JUDGE_MODEL` to a model your Groq account can access and re-run
   before quoting either number.

## Project structure

```
genai-rag-eval/
├── pipeline/                    # The RAG system itself
│   ├── __init__.py
│   ├── ingestor.py               # Hierarchical child-parent chunking → Chroma
│   ├── retriever.py              # BM25 + Vector + Cross-Encoder reranker
│   └── engine.py                 # Groq LLM + Chain-of-Thought RAG loop
├── data/
│   ├── example_standard.txt      # Sample technical spec (swap for your own .txt docs)
│   ├── test_cases.json           # 12-case suite across 4 categories (used by runner.py)
│   └── test_cases_extended.json  # Optional 50-case suite, same schema
├── src/                          # The eval + guardrails harness
│   ├── rag_system.py              # Adapter: builds the pipeline once, runs it per question
│   ├── guardrails.py              # Input + output guardrail checks
│   ├── eval_metrics.py            # LLM-as-judge + lexical overlap metrics
│   └── runner.py                  # Runs the full suite, writes results/results.json
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── results/
│   └── results.json              # generated by runner.py (gitignored)
├── chroma_db/                    # generated vector store (gitignored)
├── LLM Eval & Guardrails Dashboard.pdf   # Dashboard export from the 12-case run
├── main.py                       # Standalone demo runner (3-query sanity check)
├── requirements.txt
└── .gitignore                    # excludes .env, chroma_db/, results.json, caches
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
# Optional: override the judge model (default: llama-3.3-70b-versatile)
# GROQ_JUDGE_MODEL=your_model_here
```

> First run downloads roughly 1.5 GB of model weights for the embedding model
> and reranker. Cached locally after that.

Drop your own `.txt` documents into `/data` to replace the example spec —
the ingestor loads every `.txt` file in that folder automatically.

## Running it

```bash
# Quick sanity check — 3 demo queries, no eval/guardrails
python main.py

# Full evaluation + guardrails suite (12 cases x 2 passes)
python src/runner.py

# View results in the browser
streamlit run dashboard/app.py
```

## Key design decisions

| Choice | Rationale |
|---|---|
| `BAAI/bge-large-en-v1.5` embeddings | Strong open-source model, runs locally, no API cost |
| Hierarchical parent-child chunking | Child chunks (400 chars, 50 overlap) give precision; parent chunks (2000 chars, 200 overlap) give the LLM full surrounding detail. Sizes are in characters, not tokens |
| BM25 + Vector hybrid | BM25 catches exact keyword hits (register numbers, spec IDs); dense search catches paraphrases. BM25 indexes the full raw documents, the vector store indexes child chunks |
| `BAAI/bge-reranker-large` | Cross-encoder scores each deduplicated parent context against the query; the top 3 go to the LLM |
| `openai/gpt-oss-120b` on Groq, `temperature=0.0` | Deterministic, reproducible outputs |
| `INSUFFICIENT_LOCAL_CONTEXT` guard | Forces the LLM to admit when context is absent — the harness measures how often this actually works |
| Separate judge model | The judge runs on its own Groq model rather than grading itself |
| Pipeline built once, reused per case | Avoids reloading the embedding and reranker models for every test question |
| Guardrails run on the parsed FINAL ANSWER | The chain-of-thought scratch work isn't what a downstream system would consume — only the final answer is checked and scored |

## Known limitations

- **Small suite.** 12 cases (2 out-of-context, 5 adversarial) is enough to
  demonstrate the harness, but the rates are coarse: one case moves a rate by
  8 to 50 points.
- **Judge model availability.** The default judge model may not be available
  to every Groq account; when it isn't, all scores silently become 0 (the
  error only appears in each result's `justification`). Check that field
  before trusting any aggregate.
- **Numeric pre-filter is strict.** It compares digit strings, so a correct
  answer that writes `2,000` when the spec says `2000` is flagged as an
  unsupported claim. This happened on `norm_003`.
- **Single small source document.** The sample spec is about 600 characters,
  so every query retrieves essentially the same context. The tests exercise
  the generator and guardrails but barely stress retrieval.
- **Duplicate chunks across runs.** `Chroma.from_documents` is called against
  the same persist directory on every run, so re-running appends the same
  chunks again. The retriever's parent-context dedup hides this in the
  output, but it crowds the dense top-k. Delete `chroma_db/` before a clean
  run.
- **Heuristic guardrails.** The input checks are regex and phrase matching,
  so paraphrased attacks and unformatted PII can get through.

## Extending it

- **Grow the test set**: `data/test_cases_extended.json` has 50 cases,
  including paraphrased injections and unformatted PII that the input
  guardrails don't catch by design (13 of its 20 adversarial cases are
  caught). Copy it over `data/test_cases.json` and re-run to use it.
- **Fix the duplicates**: clear or reuse the Chroma collection in
  `ingestor.py` instead of appending on each run.
- **Stress retrieval**: add several longer `.txt` documents and test cases
  whose answers span or compete across them.
- **Load PDFs**: use `langchain_community.document_loaders.PyPDFLoader` and
  pass results into `ingestor.process_documents()`.
- **Add RAGAS**: swap the hand-rolled lexical metrics in `eval_metrics.py`
  for `ragas.metrics` if you want the industry-standard metric names.
- **Add a PII/NER model**: swap the regex PII detector in `guardrails.py`
  for `presidio` for named-entity-based detection.

## Resume framing

> • Engineered a RAG pipeline using LangChain, Groq-hosted LLMs, and local
> BGE embeddings, with a chain-of-thought prompt and an
> INSUFFICIENT_LOCAL_CONTEXT guard that makes the model decline instead of
> answering beyond the retrieved context.
>
> • Architected hybrid retrieval combining BM25 and dense vector search with
> hierarchical parent-child chunking, then re-ranked the merged candidates
> with a BGE cross-encoder so only the top-3 most relevant context blocks
> reach the LLM.
>
> • Built an evaluation and guardrails harness with LLM-as-judge scoring and
> a Streamlit dashboard; on a 12-case suite (in-scope, out-of-context,
> prompt-injection, PII), the pipeline declined 2/2 out-of-context queries
> and input guardrails blocked 5/5 injection and PII attempts before they
> reached the LLM.

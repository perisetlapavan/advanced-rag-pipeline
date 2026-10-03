"""
Evaluation metrics.

1. LLM-as-judge: scores each (question, context, answer, expected_answer) tuple
   on correctness and groundedness (1-5), with a written justification.
   This is the headline metric — it's what most eval discussions in interviews
   center on.

2. Lightweight lexical overlap metrics (fast, free, no LLM call) as a
   secondary signal: context_overlap (how much of the answer's content
   appears in context -> proxy for faithfulness) and answer_relevancy
   (token overlap between question and answer -> proxy for on-topic-ness).

Swap the judge model / add ragas.metrics (faithfulness, answer_relevancy,
context_precision, context_recall) here if you want the full RAGAS suite —
this hand-rolled version keeps the project dependency-light and easy to
explain line-by-line in an interview.
"""

import json
import os
import re
from groq import Groq

JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")
        _client = Groq(api_key=api_key)
    return _client


JUDGE_PROMPT = """You are an evaluation judge for a RAG system. Score the ANSWER
against the CONTEXT and EXPECTED_ANSWER on two dimensions, 1-5 each:

- correctness: does the answer convey the same information as expected_answer?
- groundedness: is every claim in the answer supported by the context (no
  invented facts)?

Respond with ONLY a JSON object, no other text:
{{"correctness": <1-5>, "groundedness": <1-5>, "justification": "<one sentence>"}}

CONTEXT: {context}
QUESTION: {question}
EXPECTED_ANSWER: {expected_answer}
ANSWER: {answer}
"""


def llm_judge_score(question: str, context: str, expected_answer: str, answer: str) -> dict:
    client = _get_client()
    prompt = JUDGE_PROMPT.format(
        context=context, question=question, expected_answer=expected_answer, answer=answer
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    try:
        # strip markdown fences if the model adds them despite instructions
        cleaned = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        return {
            "correctness": int(parsed.get("correctness", 0)),
            "groundedness": int(parsed.get("groundedness", 0)),
            "justification": parsed.get("justification", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {"correctness": 0, "groundedness": 0, "justification": f"PARSE_ERROR: {raw[:120]}"}


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def context_overlap_score(answer: str, context: str) -> float:
    """Fraction of answer tokens also found in context. Cheap faithfulness proxy."""
    a_tokens, c_tokens = _tokenize(answer), _tokenize(context)
    if not a_tokens:
        return 0.0
    return round(len(a_tokens & c_tokens) / len(a_tokens), 3)


def answer_relevancy_score(question: str, answer: str) -> float:
    """Token overlap between question and answer. Cheap on-topic proxy."""
    q_tokens, a_tokens = _tokenize(question), _tokenize(answer)
    if not q_tokens:
        return 0.0
    return round(len(q_tokens & a_tokens) / len(q_tokens), 3)

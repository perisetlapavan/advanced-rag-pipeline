from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from pipeline.retriever import AdvancedTechnicalRetriever


# Chain-of-Thought system prompt enforcing grounded, non-hallucinated answers
_COT_SYSTEM_PROMPT = """You are an expert enterprise automation and systems engineer.
Your task is to answer technical queries using ONLY the verified context blocks provided below.

Follow this execution loop strictly:

1. THINKING STEP:
   Analyze the context snippets for exact parameters, limits, or configuration values.
   Write down your logical deduction step by step.

2. VERIFICATION STEP:
   Cross-check whether the context explicitly supports your statement.
   Note any information gaps clearly.

3. FINAL ANSWER:
   Provide a clear, precise technical instruction or specification code block.

IMPORTANT: If the answer cannot be confidently derived from the context, respond with:
  INSUFFICIENT_LOCAL_CONTEXT
Do not hallucinate, guess, or infer beyond what is written in the context."""


class TechnicalRAGPipeline:
    """
    Groq-powered orchestration layer.
    - Retrieves grounded context via AdvancedTechnicalRetriever.
    - Applies a Chain-of-Thought prompt enforcing strict context adherence.
    - Uses llama-3.3-70b-versatile at temperature=0.0 for deterministic output.
    """

    def __init__(self, retriever: AdvancedTechnicalRetriever):
        self.retriever = retriever
        print("  [Engine] Connecting to Groq API (llama-3.3-70b-versatile)...")
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _COT_SYSTEM_PROMPT),
                ("user", "CONTEXT:\n{context}\n\nQUERY: {question}"),
            ]
        )
        self.chain = self.prompt | self.llm

    def ask(self, question: str, top_n: int = 3) -> str:
        """
        End-to-end RAG query: retrieve → format context → invoke LLM.

        Args:
            question:  Natural language technical query.
            top_n:     Number of context blocks to pass to the LLM.

        Returns:
            LLM response string.
        """
        contexts = self.retriever.retrieve(question, top_n=top_n)

        if not contexts:
            return "INSUFFICIENT_LOCAL_CONTEXT — no relevant documents found."

        formatted_context = "\n\n---\n\n".join(contexts)
        response = self.chain.invoke(
            {"context": formatted_context, "question": question}
        )
        return response.content

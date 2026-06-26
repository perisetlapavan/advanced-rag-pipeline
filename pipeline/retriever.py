import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class AdvancedTechnicalRetriever:
    """
    Hybrid retrieval stack:
      1. Dense vector search  (semantic similarity via Chroma + BGE embeddings)
      2. Sparse BM25 search   (exact keyword / token matching)
      3. Cross-encoder reranker (BAAI/bge-reranker-large) on merged candidate set
    
    Child chunks are retrieved for precision; parent context is surfaced for the LLM.
    """

    def __init__(self, vector_db: Chroma, all_documents: list[Document]):
        print("  [Retriever] Building BM25 index over document corpus...")
        self.vector_retriever = vector_db.as_retriever(search_kwargs={"k": 8})
        self.bm25_retriever = BM25Retriever.from_documents(all_documents)
        self.bm25_retriever.k = 8

        print("  [Retriever] Loading cross-encoder: BAAI/bge-reranker-large ...")
        self.rerank_model = CrossEncoder("BAAI/bge-reranker-large")

    def retrieve(self, query: str, top_n: int = 3) -> list[str]:
        """
        Run hybrid retrieval and return the top-N reranked parent context strings.

        Args:
            query:  User query string.
            top_n:  Number of final context blocks to return.

        Returns:
            List of parent context strings, ordered by reranker score descending.
        """
        dense_results = self.vector_retriever.invoke(query)
        sparse_results = self.bm25_retriever.invoke(query)

        # Deduplicate on parent_context — surface the broader block, not raw child chunks
        seen_contexts: set[str] = set()
        candidate_contexts: list[str] = []

        for doc in dense_results + sparse_results:
            parent_context = doc.metadata.get("parent_context", doc.page_content)
            if parent_context not in seen_contexts:
                seen_contexts.add(parent_context)
                candidate_contexts.append(parent_context)

        if not candidate_contexts:
            return []

        # Score every (query, context) pair with the cross-encoder
        pairs = [[query, ctx] for ctx in candidate_contexts]
        scores = self.rerank_model.predict(pairs)

        ranked_indices = np.argsort(scores)[::-1]
        top_contexts = [candidate_contexts[idx] for idx in ranked_indices[:top_n]]

        print(
            f"  [Retriever] {len(candidate_contexts)} candidates → "
            f"top {len(top_contexts)} returned after reranking."
        )
        return top_contexts

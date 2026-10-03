import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class TechnicalDocIngestor:
    """
    Hierarchical child-parent document ingestion.
    - Parent chunks (2000 tokens) preserve broad context.
    - Child chunks (400 tokens) are indexed in the vector store for precision retrieval.
    - Each child carries its parent's full text in metadata for re-ranking.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        print("  [Ingestor] Loading embedding model: BAAI/bge-large-en-v1.5 ...")
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        self.persist_directory = persist_directory

        # Large parent blocks — preserve surrounding context
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        # Small child fragments — tighter semantic precision for vector search
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50
        )

    def process_documents(self, raw_documents: list[Document]) -> Chroma:
        """
        Split raw docs into hierarchical child-parent chunks and persist to Chroma.

        Returns:
            Chroma vector store instance.
        """
        parent_docs = self.parent_splitter.split_documents(raw_documents)
        final_child_docs = []

        for i, parent_doc in enumerate(parent_docs):
            child_chunks = self.child_splitter.split_text(parent_doc.page_content)
            for child_chunk in child_chunks:
                child_doc = Document(
                    page_content=child_chunk,
                    metadata={
                        **parent_doc.metadata,
                        "parent_context": parent_doc.page_content,
                        "parent_id": f"parent_{i}",
                    },
                )
                final_child_docs.append(child_doc)

        print(
            f"  [Ingestor] Created {len(parent_docs)} parent chunks → "
            f"{len(final_child_docs)} child chunks for indexing."
        )

        vector_db = Chroma.from_documents(
            documents=final_child_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"  [Ingestor] Vector DB persisted at: {self.persist_directory}")
        return vector_db

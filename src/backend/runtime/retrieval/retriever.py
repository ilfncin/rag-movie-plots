import logging
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from backend.config.settings import (
    EMBEDDING_CONFIG,
    VECTORSTORE_CONFIG,
    RETRIEVER_CONFIG
)

logger = logging.getLogger("RETRIEVER")

class Retriever:
    """
    Loads a persisted Chroma vector store and retrieves the most relevant chunks.

    Notes:
    - This project stores *chunks* as LangChain Documents in Chroma (page_content = chunk text).
    - similarity_search_with_score returns (Document, distance) for cosine space (lower is better).
    """
    def __init__(self):
        self.top_k = RETRIEVER_CONFIG["top_k"]
        
        self.distance_threshold = RETRIEVER_CONFIG["distance_threshold"]
        self.use_threshold = RETRIEVER_CONFIG["use_threshold"]
        
        self.model_name = EMBEDDING_CONFIG["embedding_model"]
        self.persist_dir = VECTORSTORE_CONFIG["persist_dir"]
        self.collection_name = VECTORSTORE_CONFIG["collection_name"]
        
        self.embedding_function = OpenAIEmbeddings(model=self.model_name)

        self.vectordb = Chroma(
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )

        logger.info("Loading vector store: %s", self.persist_dir)
        logger.info("Embedding model: %s", self.model_name)
        logger.info(
            "top_k=%s | use_threshold=%s | distance_threshold=%s",
            self.top_k,
            self.use_threshold,
            self.distance_threshold,
        )
        logger.info(f"Vector store metadata: {self.vectordb._collection.metadata}")

    def retrieve(self, question: str) -> List[Document]:
        """
        Returns retrieved chunks as a list[Document].

        If use_threshold=True, only chunks with distance <= distance_threshold are returned.
        Distances are cosine distances in HNSW cosine space (lower is better).
        """

        chunks_with_distances = self.vectordb.similarity_search_with_score(
            question, k=self.top_k
        )

        sorted_chunks = sorted(chunks_with_distances, key=lambda x: x[1])

        self._log_distance_summary(sorted_chunks)
       
        if not self.use_threshold:
            accepted_chunks = [chunk for chunk, _ in sorted_chunks]
            self._debug_log_chunks(sorted_chunks, accepted_mask=None)
            logger.info(
                "Threshold disabled | returning top-%s chunks (no semantic filtering applied).",
                len(accepted_chunks),
            )
            return accepted_chunks
        
        accepted: List[Document] = []
        accepted_mask: List[bool] = []

        for chunk, distance in sorted_chunks:
            ok = distance <= self.distance_threshold
            accepted_mask.append(ok)
            if ok:
                accepted.append(chunk)

        self._debug_log_chunks(sorted_chunks, accepted_mask=accepted_mask)

        if not accepted:
            logger.warning(
                "Threshold enabled but no chunks passed | distance_threshold=%s | returning 0 chunks.",
                self.distance_threshold
            )
            return []
            
        self._log_accepted_summary(sorted_chunks)

        return accepted
    
    # Logging helpers

    def _log_distance_summary(self, sorted_chunks: List[Tuple[Document, float]]) -> None:
        distances_summary = [
            f"{idx}:{distance:.4f}"
            for idx, (_, distance) in enumerate(sorted_chunks, start=1)
        ]
        logger.info(
            "Retrieved %s candidates (sorted by cosine distance; lower is better): %s",
            len(sorted_chunks),
            distances_summary,
        )

    def _log_accepted_summary(
        self,
        sorted_chunks: List[Tuple[Document, float]],
    ) -> None:
        accepted_summary = [
            f"{idx}:{distance:.4f}"
            for idx, (_, distance) in enumerate(sorted_chunks, start=1)
            if distance <= self.distance_threshold
        ]

        logger.info(
            "Accepted %s chunks under threshold %s: %s",
            len(accepted_summary),
            self.distance_threshold,
            accepted_summary,
        )

    def _debug_log_chunks(
        self,
        sorted_chunks: List[Tuple[Document, float]],
        accepted_mask: List[bool] | None,
    ) -> None:
        """
        Logs chunk-level details only when DEBUG is enabled.
        accepted_mask:
            - None => threshold disabled (all accepted)
            - list[bool] => per-chunk acceptance under threshold
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        for idx, (chunk, distance) in enumerate(sorted_chunks, start=1):
            accepted = True if accepted_mask is None else bool(accepted_mask[idx - 1])
            logger.debug(self._format_chunk_debug(idx, chunk, distance, accepted))

    def _format_chunk_debug(
        self, idx: int, chunk: Document, distance: float, accepted: bool
    ) -> str:
        md = chunk.metadata or {}
        preview = (chunk.page_content or "").replace("\n", " ").strip()

        return (
            f"\nChunk {idx}"
            f"\n- Chunk ID: {md.get('chunk_id', 'N/A')}"
            f"\n- Cosine Distance: {distance:.4f} (lower is better)"
            f"\n- Accepted: {'YES' if accepted else 'NO'}"
            f"\n- Title: {md.get('Title', 'N/A')}"
            f"\n- Release Year: {md.get('Release Year', 'N/A')}"
            f"\n- Wiki Page: {md.get('Wiki Page', 'N/A')}"
            f"\n- Origin/Ethnicity: {md.get('Origin/Ethnicity', 'N/A')}"
            f"\n- Director: {md.get('Director', 'N/A')}"
            f"\n- Cast: {md.get('Cast', 'N/A')}"
            f"\n- Genre: {md.get('Genre', 'N/A')}"
            f"\n- Preview: {preview}"
        )
    
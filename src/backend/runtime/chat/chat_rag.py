import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

from backend.runtime.retrieval.retriever import Retriever
from backend.infra.llm_client import LLMClient
from backend.runtime.chat.prompt_builder import PromptBuilder


logger = logging.getLogger("CHAT_RAG")

class ChatRAG:
    """
    Online RAG chat wrapper.

    Notes:
    - The retriever returns *chunks* stored as LangChain Documents in Chroma.
    - This class formats retrieved chunks and calls the LLM.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder

    
    def run(self, question: str) -> Dict[str, Any]:
        logger.info("ChatRAG started | question=%r", question)

        chunks: List[Document] = self.retriever.retrieve(question)

        context = self._build_context(chunks)
        prompt = self.prompt_builder.build(question=question, context=context)

        self._log_context_stats(chunks, context)

        answer = self.llm_client.generate(prompt)

        return {
            "question": question,
            "context": context,
            "answer": answer,
        }

    def _build_context(self, chunks: List[Document]) -> str:
        if not chunks:
            return (
                "No relevant context was retrieved from the vector store for this question.\n"
                "If you cannot answer reliably without context, say so."
            )

        formatted_blocks = []

        for idx, chunk in enumerate(chunks, start=1):
            block = self._format_chunk_block(chunk)
            if not block:
                continue

            formatted_blocks.append(block)

        if not formatted_blocks:
            return (
                "Context retrieval returned chunks, but none could be formatted within the context budget.\n"
                "If you cannot answer reliably, say so."
            )

        return "\n\n".join(formatted_blocks)

    def _format_chunk_block(self, chunk: Document) -> str:
        md = chunk.metadata or {}

        chunk_id = md.get("chunk_id", "N/A")
        doc_id = md.get("doc_id", "N/A")
        title = md.get("Title", "N/A")
        genre = md.get("Genre", "N/A")
        year = md.get("Release Year", "N/A")
        director = md.get("Director", "N/A")
        cast = md.get("Cast", "N/A")
        origin = md.get("Origin/Ethnicity", "N/A")

        text = (chunk.page_content or "").strip()
        if not text:
            return ""

        return (
            f"chunk_id: {chunk_id}\n"
            f"doc_id: {doc_id}\n"
            f"Title: {title}\n"
            f"Release Year: {year}\n"
            f"Genre: {genre}\n"
            f"Director: {director}\n"
            f"Cast: {cast}\n"
            f"Origin/Ethnicity: {origin}\n"
            f"Content (Plot):\n"
            f"<<<\n{text}\n>>>"
        )

    # Logging helpers

    def _log_context_stats(self, chunks: List[Document], context: str) -> None:
        titles = []
        for c in chunks:
            md = c.metadata or {}
            title = md.get("Title") or "<missing title>"
            year = md.get("Release Year") or "<missing year>"
            titles.append(f"{title} ({year})")

        logger.info(
            "Retrieved %s chunks | context_chars=%s | sample_titles=%s",
            len(chunks),
            len(context),
            titles,
        )

        preview = context[:800].replace("\n", " ").strip()
        logger.debug("Context preview (first 800 chars): %s", preview)
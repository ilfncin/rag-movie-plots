import json
from  pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from backend.config.settings import (
    EMBEDDING_CONFIG,
    VECTORSTORE_CONFIG
)

import logging

logger = logging.getLogger("VECTORSTORE")


class VectorStorePipeline:
    """
    Builds a ChromaDB vector store from preprocessed text chunks.

    The pipeline loads chunked documents from JSONL files, generates 
    embeddings using OpenAIEmbeddings (text-embedding-3-small), and 
    persists the resulting vectors locally.
    """
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.model_name = EMBEDDING_CONFIG["embedding_model"]
        self.persist_dir = VECTORSTORE_CONFIG["persist_dir"]
        self.collection_name = VECTORSTORE_CONFIG["collection_name"]
        self.embedding_function = OpenAIEmbeddings(model=self.model_name)

    def run(self):

        logger.info(f"Reading chunks: {self.input_path}")

        with open(self.input_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Embedding model: {self.model_name}")
        logger.info(f"Persist directory: {self.persist_dir}")

        documents = []
        ids = []

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                )
            )
            ids.append(chunk["chunk_id"])

        Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            ids=ids,
            collection_metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vectorstore created successfully!")
import json
import pathlib

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.config import (
    EMBEDDING_CONFIG,
    VECTORSTORE_CONFIG
)


class VectorStorePipeline:
    def __init__(self, input_path: pathlib.Path):
        self.input_path = input_path
        self.model_name = EMBEDDING_CONFIG["embedding_model"]
        self.persist_dir = VECTORSTORE_CONFIG["persist_dir"]
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.model_name)

    def run(self):

        print(f"Reading chunks from {self.input_path}...")
        with open(self.input_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        print(f"Preparing {len(chunks)} documents for vector storage...")
        documents = [
            Document(
                page_content=chunk["text"],
                metadata={**chunk["metadata"], "source_id": chunk["id"]}
            )
            for chunk in chunks
        ]

        print(f"Generating embeddings with model: {self.model_name}")
        Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_dir
        )
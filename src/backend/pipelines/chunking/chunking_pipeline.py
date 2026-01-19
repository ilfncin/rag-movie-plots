import json
from pathlib import Path
from backend.config.settings import CHUNKING_CONFIG
from backend.pipelines.chunking.chunk_strategy import (
    CharacterSplitter, TokenSplitter, RecursiveSplitter
)

import logging

logger = logging.getLogger("CHUNKING")


class ChunkingPipeline:
    """
    Splits text documents into smaller chunks based on the configured strategy
    (character, token, or recursive) for downstream embedding and retrieval.

    Reads a JSONL file, applies the chosen splitter, and saves the resulting
    chunks as a new JSONL file.
    """
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.strategy = CHUNKING_CONFIG["strategy"]
        self.chunk_size = CHUNKING_CONFIG["chunk_size"]
        self.chunk_overlap = CHUNKING_CONFIG["chunk_overlap"]
        self.splitter = self._get_splitter()

    def _get_splitter(self):
        match self.strategy:
            case "character":
                return CharacterSplitter(self.chunk_size, self.chunk_overlap)
            case "token":
                return TokenSplitter(self.chunk_size, self.chunk_overlap)
            case "recursive":
                return RecursiveSplitter(self.chunk_size, self.chunk_overlap)
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")

    def run(self):
        logger.info(f"Strategy: {self.strategy}")
        logger.info(f"Chunk size: {self.chunk_size} | Overlap: {self.chunk_overlap}")

        with open(self.input_path, "r", encoding="utf-8") as f:
            documents = [json.loads(line) for line in f]

        logger.info(f"Total documents: {len(documents)}")

        chunks = []
        for doc in documents:
            doc_id = doc["id"]
            metadata = doc["metadata"]

            doc_chunks = self.splitter.split(doc["text"])

            for i, chunk_text in enumerate(doc_chunks):
                chunk_id = f"{doc_id}_{i}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "chunk_index": i
                    }
                })

        logger.info(f"Total chunks generated: {len(chunks)}")
        logger.info(f"Saving to {self.output_path}")

        with open(self.output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
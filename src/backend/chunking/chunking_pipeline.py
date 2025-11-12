from backend.config import CHUNKING_CONFIG
from backend.chunking.chunk_strategy import (
    CharacterSplitter, TokenSplitter, RecursiveSplitter
)
import json
from pathlib import Path

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
        with open(self.input_path, "r", encoding="utf-8") as f:
            documents = [json.loads(line) for line in f]

        chunks = []
        for doc in documents:
            doc_chunks = self.splitter.split(doc["text"])
            for chunk in doc_chunks:
                chunks.append({
                    "id": doc["id"],
                    "text": chunk,
                    "metadata": doc.get("metadata", {})
                })

        print(f"Saving {len(chunks)} chunks to {self.output_path}")
        with open(self.output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
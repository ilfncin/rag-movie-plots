from backend.config import RAG_CONFIG
from backend.chunking.chunk_strategy import (
    CharacterSplitter, TokenSplitter, RecursiveSplitter
)
import json
from pathlib import Path

class ChunkingPipeline:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.config = RAG_CONFIG["chunking"]
        self.strategy = self._get_splitter()

    def _get_splitter(self):
        strategy = self.config["strategy"]
        chunk_size = self.config["chunk_size"]
        chunk_overlap = self.config["chunk_overlap"]

        match strategy:
            case "character":
                return CharacterSplitter(chunk_size, chunk_overlap)
            case "token":
                return TokenSplitter(chunk_size, chunk_overlap)
            case "recursive":
                return RecursiveSplitter(chunk_size, chunk_overlap)
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            documents = [json.loads(line) for line in f]

        chunks = []
        for doc in documents:
            doc_chunks = self.strategy.split(doc["text"])
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
from typing import List
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from backend.pipelines.chunking.chunk_strategy import ChunkStrategy


class RecursiveSplitter(ChunkStrategy):
    def __init__(self, chunk_size=1200, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""]
        )


    def split(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
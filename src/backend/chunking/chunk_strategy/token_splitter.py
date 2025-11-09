from typing import List
from langchain_text_splitters import (
    TokenTextSplitter
)

from backend.chunking.chunk_strategy.chunk_strategy import ChunkStrategy

class TokenSplitter(ChunkStrategy):
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
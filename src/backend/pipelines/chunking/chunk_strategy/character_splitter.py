from langchain_text_splitters import CharacterTextSplitter
from .chunk_strategy import ChunkStrategy


class CharacterSplitter(ChunkStrategy):
    def __init__(self, chunk_size=1200, chunk_overlap=200):
        self.splitter = CharacterTextSplitter(
            separator=". ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split(self, text: str):
        return self.splitter.split_text(text)
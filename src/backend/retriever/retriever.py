from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import (
    EMBEDDING_CONFIG,
    VECTORSTORE_CONFIG,
    RETRIEVER_CONFIG
)

class Retriever:
    """
    This class is responsible for loading a persisted Chroma vector store
    and returning a retriever object that can be used to query the vector database.

    The embeddings must match those used during ingestion.
    """
    def __init__(self, top_k):
        self.top_k = top_k or RETRIEVER_CONFIG["top_k"]
        self.model_name = EMBEDDING_CONFIG["embedding_model"]
        self.persist_dir = VECTORSTORE_CONFIG["persist_dir"]
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.model_name)

    def load(self):
        print(f"Loading vector store from: {self.persist_dir}")
        print(f"Using embedding model: {self.model_name}")
        print(f"Retrieval: top_k={self.top_k}")
        
        vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_function
        )
        retriever = vectordb.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": self.top_k}
        )
        print("Retriever loaded successfully.")
        return retriever

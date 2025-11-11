import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Data Cleaning Configuration
CLEANING_CONFIG: Dict[str, Any] = {
    "invalid_values": ["", "unknown", "unk", "nan", "none", "null"],
    "exceptions": {
        "Title": {"Unknown"}  # 'Unknown' is valid only in Title column
    },
    "fill_text": "Not specified",
    "columns": [
        "Title", "Plot", "Genre", "Release Year",
        "Director", "Cast", "Origin/Ethnicity", "Wiki Page"
    ],
    "text_column": "Plot"
}

# Chunking Configuration
CHUNKING_CONFIG: Dict[str, Any] = {
    "strategy": os.getenv("CHUNK_STRATEGY", "recursive"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50))
}

# Embedding Configuration
EMBEDDING_CONFIG: Dict[str, Any] = {
    "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
}

# Vectorstore Configuration
VECTORSTORE_CONFIG: Dict[str, Any] = {
    "persist_dir": os.getenv("PERSIST_DIR", "db/chroma"),
}

# Retriever Configuration
RETRIEVER_CONFIG: Dict[str, Any] = {
    "top_k": int(os.getenv("RETRIEVER_TOP_K", 5)),
}

# LLM (OpenAI) Configuration
LLM_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": os.getenv("LLM_MODEL", "gpt-4o"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", 0.0)),
}


import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

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

CHUNKING_CONFIG: Dict[str, Any] = {
    "strategy": os.getenv("CHUNK_STRATEGY", "recursive"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50))
}

EMBEDDING_CONFIG: Dict[str, Any] = {
    "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
}

VECTORSTORE_CONFIG: Dict[str, Any] = {
    "persist_dir": os.getenv("PERSIST_DIR", "db/chroma"),
}

OPENAI_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
}

RAG_CONFIG: Dict[str, Any] = {
    "cleaning": CLEANING_CONFIG,
    "chunking": CHUNKING_CONFIG,
    "embedding": EMBEDDING_CONFIG,
    "vectorstore": VECTORSTORE_CONFIG,
    "openai": OPENAI_CONFIG
}

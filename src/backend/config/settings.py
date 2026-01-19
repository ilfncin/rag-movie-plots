from pathlib import Path
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}

# Data Cleaning Configuration
CLEANING_CONFIG: Dict[str, Any] = {
    "invalid_values": ["", "unknown", "unk", "none", "null"],
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
    "chunk_size": int(os.getenv("CHUNK_SIZE", 1200)),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200))
}

# Embedding Configuration
EMBEDDING_CONFIG: Dict[str, Any] = {
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
}

# Vectorstore Configuration

# Resolve the project root directory.
# NOTE: This assumes the current file lives at:
# <project_root>/src/backend/config/settings.py
# If the folder structure changes, this index MUST be updated.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VECTORSTORE_CONFIG = {
    "persist_dir": str((PROJECT_ROOT / os.getenv("PERSIST_DIR", "db/chroma")).resolve()),
    "collection_name": os.getenv("VECTORSTORE_COLLECTION_NAME", "movie_plots")
}

# Retriever Configuration
RETRIEVER_CONFIG: Dict[str, Any] = {
    "top_k": int(os.getenv("RETRIEVER_TOP_K", 10)),
    "use_threshold": _env_bool("RETRIEVER_USE_THRESHOLD", default=False),
    "distance_threshold": float(os.getenv("RETRIEVER_DISTANCE_THRESHOLD", 0.35))
}

# LLM (OpenAI) Configuration
LLM_CONFIG: Dict[str, Any] = {
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", 0.0))
}


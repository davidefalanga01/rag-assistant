import os

from dotenv import load_dotenv


load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


NEW_EMBED_MODEL = env_bool("NEW_EMBED_MODEL", True)
ENHANCED_CHUNKING = env_bool("ENHANCED_CHUNKING", False)

USE_RERANKER = env_bool("USE_RERANKER", False)
USE_ADAPTIVE_K = env_bool("USE_ADAPTIVE_K", False)
USE_METADATA_FILTERING = env_bool("USE_METADATA_FILTERING", False)
USE_HYBRID_SEARCH = env_bool("USE_HYBRID_SEARCH", False)

DB_PATH = os.getenv("DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection_simple")

EMBED_MODEL = (
    "BAAI/bge-large-en-v1.5"
    if NEW_EMBED_MODEL
    else "sentence-transformers/all-MiniLM-L6-v2"
)

FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))
CANDIDATE_TOP_K = int(os.getenv("CANDIDATE_TOP_K", "20"))
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.60"))

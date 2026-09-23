from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

PDF_PATH = ROOT / "data" / "data.pdf"
INDEX_PATH = ROOT / "artifacts" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

MAX_RETRIES = 3       

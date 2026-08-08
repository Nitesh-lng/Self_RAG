'''question — the user query; may be transformed for re-retrieval
documents — retrieved chunks; filtered after relevance grading
generation — the LLM's generated answer; updated on each regeneration
retries — retry counter; incremented on every regenerate/re-retrieve and capped at MAX_RETRIES (defined in config)'''

from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int
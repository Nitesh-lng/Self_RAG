class SelfRAGError(Exception):
    """Base exception for the Self-RAG project."""
    pass


class DocumentLoadError(SelfRAGError):
    """Raised when document loading fails."""
    pass


class VectorStoreError(SelfRAGError):
    """Raised when vector store build/load fails."""
    pass
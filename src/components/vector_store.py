from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from src.config.settings import EMBEDDING_MODEL


class VectorStoreBuilder:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

    def build(self, chunks: list[Document]) -> FAISS:
        return FAISS.from_documents(
            chunks,
            self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )

    def save(self, vector_store, path):
        vector_store.save_local(str(path))

    def load(self, path):
        return FAISS.load_local(
            str(path), self.embeddings, allow_dangerous_deserialization=True
        )
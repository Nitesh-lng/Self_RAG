from pathlib import Path
from src.config.settings import PDF_PATH, INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.components.loader import PDFDocumentLoader   
from src.components.chunker import TextSplitter       
from src.components.vector_store import VectorStoreBuilder
from src.pipeline.graph import app
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    builder = VectorStoreBuilder()

    if Path(INDEX_PATH).exists():
        logger.info("Loading existing vector store...")
        builder.load(INDEX_PATH)
    else:
        logger.info("Building vector store...")
        loader = PDFDocumentLoader(PDF_PATH)
        documents = loader.load()
        splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.chunk(documents)
        vector_store = builder.build(chunks)
        builder.save(vector_store, INDEX_PATH)
        logger.info("Vector store saved.")

    print("-" * 50)

    while True:
        query = input("\nEnter your query (or type 'exit'): ")
        if query == "exit":
            break

        result = app.invoke({"question": query, "retries": 0})

        print(f"\nQuestion: {query}")
        print(f"Answer: {result['generation']}")


if __name__ == "__main__":
    main()
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from src.utils.exceptions import DocumentLoadError


class PDFDocumentLoader:
    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)

    def load(self):
        if not self.pdf_path.exists():
            raise DocumentLoadError(f"PDF not found: {self.pdf_path}")
        if self.pdf_path.suffix.lower() != ".pdf":
            raise DocumentLoadError(f"Wrong file format: {self.pdf_path}")

        loader = PyPDFLoader(str(self.pdf_path))
        documents = loader.load()

        if not documents:
            raise DocumentLoadError(f"File is blank/scanned: {self.pdf_path}")

        return documents
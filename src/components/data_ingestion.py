import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List

class PDFIngestion:
    """
    Handles the loading of PDF documents from a specified directory.
    """
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_documents(self) -> List[Document]:
        """
        Process all PDF files in the directory and return a list of LangChain Documents.
        """
        all_documents = []
        
        # Check if directory exists
        if not self.data_dir.exists():
            print(f"Directory {self.data_dir} does not exist.")
            return []

        # Find all PDF files recursively
        pdf_files = list(self.data_dir.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process in {self.data_dir}")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Add source information to metadata
                for doc in documents:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'
                
                all_documents.extend(documents)
                print(f"  ✓ Loaded {len(documents)} pages")
                
            except Exception as e:
                print(f"  ✗ Error loading {pdf_file.name}: {e}")
        
        print(f"Total documents loaded: {len(all_documents)}")
        return all_documents

if __name__ == "__main__":
    # Simple test to run this file directly
    ingestor = PDFIngestion("data")
    docs = ingestor.load_documents()
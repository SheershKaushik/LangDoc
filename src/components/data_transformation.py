from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class DataTransformation:
    """
    Handles the transformation of raw documents into chunks.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        """
        print(f"Splitting {len(documents)} documents...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f"  ✓ Split into {len(split_docs)} chunks")
        
        if split_docs:
            print(f"  Example chunk: {split_docs[0].page_content[:100]}...")
            
        return split_docs
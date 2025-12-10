from src.components.data_ingestion import PDFIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_handler import EmbeddingManager, VectorStore

class IngestionPipeline:
    """
    Orchestrates the data ingestion process: Load -> Split -> Embed -> Store.
    """
    def __init__(self, data_dir: str, persist_dir: str = "vector_store_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.ingestion = PDFIngestion(data_dir)
        self.transformation = DataTransformation()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(self.embedding_manager.get_embedding_dim())

    def run_pipeline(self):
        print("--- Starting Ingestion Pipeline ---")
        
        documents = self.ingestion.load_documents()
        if not documents:
            print("No documents found.")
            return None

        chunks = self.transformation.split_documents(documents)
        
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        
        self.vector_store.add_documents(chunks, embeddings)
        
        # Save to disk
        self.vector_store.save_local(self.persist_dir)
        
        print("--- Ingestion Pipeline Completed & Saved ---")
        return self.vector_store, self.embedding_manager
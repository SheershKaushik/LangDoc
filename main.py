import os
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.components.model_handler import VectorStore, EmbeddingManager

def main():
    PERSIST_DIR = "vector_store_db"
    DATA_DIR = "data"
    
    # Initialize Embedding Manager (needed for both paths)
    embedding_manager = EmbeddingManager()
    embedding_dim = embedding_manager.get_embedding_dim()
    
    vector_store = None

    # Check if vector store exists
    if os.path.exists(os.path.join(PERSIST_DIR, "index.faiss")):
        print(">>> Found existing Vector Store. Loading...")
        try:
            vector_store = VectorStore(embedding_dim)
            vector_store.load_local(PERSIST_DIR)
        except Exception as e:
            print(f"Error loading vector store: {e}. Re-running ingestion.")
    
    # If not loaded, run ingestion
    if not vector_store:
        print(">>> No Vector Store found. Starting Ingestion...")
        ingestion = IngestionPipeline(DATA_DIR, PERSIST_DIR)
        vector_store, _ = ingestion.run_pipeline()
    
    if not vector_store:
        print("Failed to initialize system.")
        return

    # Initialize Prediction Pipeline
    rag_pipeline = PredictionPipeline(vector_store, embedding_manager)
    
    # Interactive Loop
    print("\nSystem Ready! Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your question: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        response = rag_pipeline.run_pipeline(user_query)
        print(f"\nAnswer:\n{response}")

if __name__ == "__main__":
    main()
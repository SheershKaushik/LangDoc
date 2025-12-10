from src.components.model_handler import RAGRetriever, OllamaLLM

class PredictionPipeline:
    """
    Orchestrates the RAG flow: Query -> Retrieve -> Generate.
    """
    def __init__(self, vector_store, embedding_manager):
        # We need the populated vector_store from the ingestion pipeline
        self.retriever = RAGRetriever(vector_store, embedding_manager)
        self.llm = OllamaLLM()

    def run_pipeline(self, query: str):
        """
        Runs the RAG prediction pipeline.
        """
        print(f"\n--- Processing Query: {query} ---")
        
        # 1. Retrieve Context
        # You can adjust top_k and threshold here
        retrieved_docs = self.retriever.retrieve(query, top_k=9, score_threshold=0.6)
        
        if not retrieved_docs:
            return "No relevant context found to answer the question."

        # Prepare context string
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # 2. Generate Answer
        answer = self.llm.generate_response(query, context)
        
        return answer
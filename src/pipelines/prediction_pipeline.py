from typing import Dict, Any, List, Optional

from src.components.model_handler import RAGRetriever, OllamaLLM


class PredictionPipeline:
    """
    Orchestrates the RAG flow: Query -> Retrieve -> Generate.

    Enhanced to return a structured result similar to the notebook's AdvancedRAGPipeline:
    returns a dict with keys: question, answer, sources, confidence, summary, context (optional).
    Also keeps a history list on the instance.
    """

    def __init__(self, vector_store, embedding_manager):
        # We need the populated vector_store from the ingestion pipeline
        self.retriever = RAGRetriever(vector_store, embedding_manager)
        self.llm = OllamaLLM()
        self.history: List[Dict[str, Any]] = []

    def run_pipeline(
        self,
        query: str,
        top_k: int = 16,
        score_threshold: float = 0.4,
        summarize: bool = False,
        return_context: bool = False,
        max_tokens: int | None = None,
        preview_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs the RAG prediction pipeline and returns structured output.

        Args:
            query: user question
            top_k: number of retrieved docs
            score_threshold: minimum similarity score to include a doc
            summarize: whether to produce a short summary of the answer
            return_context: whether to include the concatenated retrieved context

        Returns:
            dict containing question, answer, sources (list), confidence (float), summary (optional), context (optional)
        """
        print(f"\n--- Processing Query: {query} ---")

        # 1. Retrieve Context
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)

        if not retrieved_docs:
            result: Dict[str, Any] = {
                'question': query,
                'answer': "No relevant context found to answer the question.",
                'sources': [],
                'confidence': 0.0,
                'summary': None,
                'context': "" if return_context else None,
            }
            self.history.append(result)
            return result

        # Prepare context string
        context = "\n\n".join([doc.get('content', '') for doc in retrieved_docs])

        # 2. Generate Answer (allow overriding generation length via max_tokens)
        answer = self.llm.generate_response(query, context, num_predict=max_tokens)

        # 3. Build sources list and compute confidence
        sources: List[Dict[str, Any]] = []
        scores: List[float] = []

        for doc in retrieved_docs:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {}) or {}
            source = metadata.get('source_file') or metadata.get('source') or metadata.get('filename') or 'unknown'
            page = metadata.get('page') or metadata.get('page_number') or metadata.get('page_index') or 'unknown'
            score = doc.get('score') or doc.get('similarity_score') or None
            # preview_length=None means return full content; otherwise trim to preview_length
            if preview_length is None:
                preview = content
            else:
                preview = content[:preview_length]
            try:
                score_val = float(score) if score is not None else None
            except Exception:
                score_val = None

            if score_val is not None:
                scores.append(score_val)

            sources.append({'source': source, 'page': page, 'score': score_val, 'preview': preview})

        confidence = max(scores) if scores else 0.0

        # 4. Optional summarization
        summary: Optional[str] = None
        if summarize and answer:
            try:
                # Use a simple summarization call. This uses the LLM to summarize the final answer.
                summary = self.llm.generate_response_simple("Summarize the following answer in 2 sentences.", answer)
            except Exception as e:
                summary = f"Error generating summary: {e}"

        result = {
            'question': query,
            'answer': answer,
            'sources': sources,
            'confidence': float(confidence),
            'summary': summary,
            'context': context if return_context else None,
        }

        # 5. store history and return
        self.history.append(result)
        return result
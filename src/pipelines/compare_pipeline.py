import os
import json
from typing import List, Dict, Any, Optional

from src.components.model_handler import RAGRetriever, OllamaLLM


class ComparePipeline:
    """Compare two or more documents on a topic and return similarities/differences."""

    def __init__(self, vector_store, embedding_manager):
        self.retriever = RAGRetriever(vector_store, embedding_manager)
        self.llm = OllamaLLM()

    def _collect_document_context(self, doc_id: str, topic: str, top_k: int) -> Dict[str, Any]:
        """Retrieve the most relevant chunks for a given document and topic."""
        # doc_id is a path relative to DATA_DIR (as used by main.list_documents)
        filename = os.path.basename(doc_id)

        retrieved = self.retriever.retrieve(topic, top_k=top_k, score_threshold=0.0)
        # Filter to entries that appear to come from this file
        filtered = [r for r in retrieved if (r.get('metadata') or {}).get('source_file') == filename or (r.get('metadata') or {}).get('source') == filename]

        # If the retriever didn't tag source_file, fallback to any metadata that mentions filename
        if not filtered:
            filtered = [r for r in retrieved if filename in str((r.get('metadata') or {}).get('source', ''))]

        # Build text preview and combined context
        previews = [r.get('content','')[:800] for r in filtered]
        context = "\n\n".join([r.get('content','') for r in filtered])

        return {
            'doc_id': doc_id,
            'filename': filename,
            'previews': previews,
            'context': context,
            'raw_retrieved': filtered,
        }

    def compare(self, doc_ids: List[str], topic: str, top_k: int = 8, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Run comparison across the provided doc_ids for the given topic.

        Returns a dict with keys: topic, summary, items (list per document)
        """
        if not doc_ids or len(doc_ids) < 2:
            return {'error': 'Provide at least two document ids to compare.'}

        docs_info = [self._collect_document_context(d, topic, top_k) for d in doc_ids]

        # Create a comparison prompt for the LLM
        prompt_parts = [f"Compare these documents on the topic: {topic}.\n\n"]
        for info in docs_info:
            prompt_parts.append(f"Document: {info['filename']}\nExtracted snippets:\n")
            for i, p in enumerate(info['previews'][:3], start=1):
                prompt_parts.append(f"Snippet {i}: {p}\n")
            prompt_parts.append("\n")

        # Request a strict JSON response from the LLM so the frontend can parse reliably
        prompt_parts.append(
            "Produce a STRICT JSON object (no surrounding text) with the following schema:\n"
            "{\n  \"similarities\": [string],\n  \"differences\": [string],\n  \"documents\": [\n    {\n      \"document_id\": string,\n      \"document_title\": string,\n      \"clauses\": [ { \"text\": string, \"explanation\": string } ]\n    }\n  ]\n}\n"
            "Only output valid JSON. If you cannot follow the schema, output a JSON object {\"error\": \"reason\"}."
        )

        full_prompt = "\n".join(prompt_parts)

        # Call the LLM with optional max_tokens
        llm_out = self.llm.generate_response(full_prompt, "", num_predict=max_tokens)

        # Try to parse JSON output from the LLM
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(llm_out)
        except Exception:
            # If parsing fails, return a fallback structure containing raw LLM text and lightweight items
            items = []
            for info in docs_info:
                key_clause_text = info['previews'][0] if info['previews'] else ''
                items.append({
                    'document_id': info['doc_id'],
                    'document_title': info['filename'],
                    'clauses': [ { 'text': key_clause_text, 'explanation': '' } ]
                })

            return {
                'topic': topic,
                'error': 'LLM did not return valid JSON',
                'raw': llm_out,
                'documents': items,
            }

        # Ensure the parsed object contains expected keys
        result: Dict[str, Any] = {
            'topic': topic,
            'similarities': parsed.get('similarities'),
            'differences': parsed.get('differences'),
            'documents': parsed.get('documents'),
        }

        return result

import numpy as np
import faiss
import uuid
import pickle
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Dimension: {self._embedding_dim}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True).astype('float32')

class FaissDocStore:
    """In-memory store for document text and metadata."""
    def __init__(self):
        self.store = {}
        self.next_index = 0
        
    def add(self, text: str, metadata: Dict[str, Any], original_id: str) -> int:
        current_index = self.next_index
        self.store[current_index] = {
            'text': text, 
            'metadata': metadata, 
            'original_id': original_id
        }
        self.next_index += 1
        return current_index
        
    def get_by_index(self, index: int) -> Dict[str, Any]:
        return self.store.get(index)

    def count(self) -> int:
        return len(self.store)

class VectorStore:
    """Manages document embeddings in a Faiss vector store with persistence."""
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.docstore = FaissDocStore()
        print(f"Vector store initialized. Dim: {self.embedding_dim}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Count mismatch between docs and embeddings")
            
        print(f"Adding {len(documents)} documents to vector store...")
        self.index.add(embeddings)
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            self.docstore.add(doc.page_content, doc.metadata, doc_id)
            
        print(f"Total documents: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        return D[0], I[0]

    def save_local(self, folder_path: str):
        """Saves the FAISS index and DocStore to disk."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save FAISS index
        index_path = os.path.join(folder_path, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save DocStore (Metadata)
        store_path = os.path.join(folder_path, "docstore.pkl")
        with open(store_path, "wb") as f:
            pickle.dump(self.docstore, f)
            
        print(f"Vector Store saved to {folder_path}")

    def load_local(self, folder_path: str):
        """Loads the FAISS index and DocStore from disk."""
        index_path = os.path.join(folder_path, "index.faiss")
        store_path = os.path.join(folder_path, "docstore.pkl")

        if not os.path.exists(index_path) or not os.path.exists(store_path):
            raise FileNotFoundError("Vector store files not found.")

        self.index = faiss.read_index(index_path)
        with open(store_path, "rb") as f:
            self.docstore = pickle.load(f)
            
        print(f"Vector Store loaded from {folder_path}. Total docs: {self.index.ntotal}")

class RAGRetriever:
    """Handles query-based retrieval."""
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        query_embedding = self.embedding_manager.generate_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.vector_store.search(query_embedding, top_k)
        retrieved_docs = []
        
        for index, distance in zip(indices, distances):
            if index == -1: continue
            
            doc_data = self.vector_store.docstore.get_by_index(index)
            if doc_data:
                # Convert L2 distance to Similarity (Approximate)
                similarity = 1.0 - (np.clip(distance**2, 0, 2) / 2.0)
                
                if similarity >= score_threshold:
                    retrieved_docs.append({
                        'content': doc_data['text'],
                        'metadata': doc_data['metadata'],
                        'score': float(similarity)
                    })
        return retrieved_docs

class OllamaLLM:
    """Handles interaction with Ollama."""
    def __init__(self, model_name: str = "deepseek-r1:8b", host: str = "http://localhost:11434"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            num_predict=1024,
            base_url=host
        )

    def generate_response(self, query: str, context: str) -> str:
        prompt_template = """### Instruction
        Analyze the context and answer the user's question.
        
        ### Context
        {context}

        ### Question
        {question}

        ### Answer
        """
        formatted_prompt = prompt_template.format(context=context, question=query)
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        return response.content
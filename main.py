import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.pipelines.compare_pipeline import ComparePipeline
import time
from src.components.model_handler import VectorStore, EmbeddingManager
from langchain_community.document_loaders import PyMuPDFLoader

# NOTE: The original snippet used `Features()` which is undefined in the repo.
# I'll use a placeholder dict here. If you have a `Features` class, point me to it and
# I'll wire it in instead.
features = {}

app = FastAPI()

# Serve UI from the `static` directory (expects `static/three.html` to exist)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals to be initialized on startup
embedding_manager: EmbeddingManager | None = None
vector_store: VectorStore | None = None
rag_pipeline: PredictionPipeline | None = None
compare_pipeline: ComparePipeline | None = None
# simple in-memory store for assessments (id, query, created_at, short_answer, pipeline_meta)
assessments_store: list[dict] = []

PERSIST_DIR = "vector_store_db"
DATA_DIR = "data"


@app.on_event("startup")
def app_startup():
    global embedding_manager, vector_store, rag_pipeline, compare_pipeline

    print("Starting application and initializing models/vector store...")

    # Initialize Embedding Manager
    embedding_manager = EmbeddingManager()
    embedding_dim = embedding_manager.get_embedding_dim()

    # Try to load existing vector store
    vector_store = None
    try:
        index_path = os.path.join(PERSIST_DIR, "index.faiss")
        if os.path.exists(index_path):
            print(">>> Found existing Vector Store. Loading...")
            vector_store = VectorStore(embedding_dim)
            vector_store.load_local(PERSIST_DIR)
    except Exception as e:
        print(f"Error loading vector store: {e}. Will attempt re-ingestion.")

    # If not loaded, run ingestion pipeline
    if not vector_store:
        print(">>> No Vector Store found or load failed. Starting Ingestion...")
        ingestion = IngestionPipeline(DATA_DIR, PERSIST_DIR)
        vector_store, _ = ingestion.run_pipeline()

    if not vector_store:
        raise RuntimeError("Failed to initialize vector store during startup")

    # Initialize prediction pipeline
    rag_pipeline = PredictionPipeline(vector_store, embedding_manager)
    # Initialize compare pipeline (uses the same vector store and embedding manager)
    compare_pipeline = ComparePipeline(vector_store, embedding_manager)
    print("Startup complete. RAG pipeline ready.")


@app.get("/")
def serve_ui():
    # Return the static HTML UI
    return FileResponse("static/three.html")


@app.get("/api/health")
def health():
    return {"status": "connected"}


@app.get("/api/documents")
def list_documents():
    """Return a list of documents found under DATA_DIR."""
    docs = []
    try:
        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                if f.lower().endswith('.pdf'):
                    full = os.path.join(root, f)
                    stat = os.stat(full)
                    docs.append({
                        'id': os.path.relpath(full, DATA_DIR),
                        'title': f,
                        'filename': f,
                        'doc_type': 'pdf',
                        'uploaded_at': int(stat.st_mtime * 1000)
                    })
    except Exception as e:
        print(f"Error listing documents: {e}")
    return docs


@app.get('/api/documents/{doc_id}/pages')
def get_document_pages(doc_id: str):
    """Load the specified PDF and return page-level text for the UI.

    doc_id is a path relative to DATA_DIR as returned by /api/documents.
    """
    path = os.path.join(DATA_DIR, doc_id)
    if not os.path.exists(path):
        return JSONResponse([], status_code=404)

    try:
        loader = PyMuPDFLoader(str(path))
        documents = loader.load()
        pages = []
        for i, doc in enumerate(documents, start=1):
            pages.append({
                'page_number': i,
                'text': doc.page_content,
                'ocr_confidence': doc.metadata.get('ocr_confidence', 1.0) if isinstance(doc.metadata, dict) else 1.0
            })
        return pages
    except Exception as e:
        print(f"Error loading PDF pages for {path}: {e}")
        return JSONResponse([], status_code=500)


@app.get('/api/documents/{doc_id}/pages/{page_number}/clauses')
@app.get("/api/assessments")
def get_assessments():
    """Return stored assessments."""
    return assessments_store
def get_page_clauses(doc_id: str, page_number: int):
    """Stub: return empty list of clauses for now."""
    return []


@app.post("/api/assessments")
async def run_assessment(request: Request):
    global rag_pipeline
    data = await request.json()
    query = data.get("query")
    document_id = data.get("document_id")

    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    print(f"[User Query] {query}")

    if rag_pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)

    # Run the pipeline synchronously (the pipeline is a normal blocking function)
    try:
        # Extract optional parameters for answer length and preview sizing
        max_tokens = data.get('max_tokens')
        preview_length = data.get('preview_length')  # None means send full preview

        # Run the main pipeline to get a structured result (dict)
        pipeline_result = rag_pipeline.run_pipeline(
            query,
            top_k=9,
            score_threshold=0.3,
            summarize=False,
            return_context=False,
            max_tokens=max_tokens,
            preview_length=preview_length,
        )

        # pipeline_result is expected to be a dict with keys: question, answer, sources, confidence, summary, context
        answer = pipeline_result.get('answer') if isinstance(pipeline_result, dict) else str(pipeline_result)

        # Normalize sources -> evidence_docs for the frontend
        evidence_docs = []
        sources = pipeline_result.get('sources') if isinstance(pipeline_result, dict) else []
        for s in sources:
            evidence_docs.append({
                'source': s.get('source', 'unknown'),
                'score': s.get('score'),
                'preview': s.get('preview', '')
            })

        # detailed: format the evidence snippets for the UI's "Details"
        detailed = []
        for ev in evidence_docs:
            src = ev.get('source', 'unknown')
            score = ev.get('score')
            preview = ev.get('preview', '')
            score_str = f"{score:.3f}" if isinstance(score, (float, int)) else str(score)
            detailed.append(f"Source: {src} (score: {score_str})\n{preview}")

        import uuid
        assessment_id = uuid.uuid4().hex

        resp = {
            'assessment_id': assessment_id,
            'short_answer': answer,
            'detailed_explanation': detailed,
            'evidence': evidence_docs,
            # keep the raw pipeline_result for debugging if needed (not required by UI)
            'pipeline_meta': {
                'confidence': pipeline_result.get('confidence') if isinstance(pipeline_result, dict) else None,
                'summary': pipeline_result.get('summary') if isinstance(pipeline_result, dict) else None
            }
        }

        return JSONResponse(resp)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post('/api/compare')
async def run_compare(request: Request):
    global compare_pipeline
    data = await request.json()
    doc_ids = data.get('doc_ids') or []
    topic = data.get('topic') or ''
    max_tokens = data.get('max_tokens')

    if not doc_ids or len(doc_ids) < 2:
        return JSONResponse({'error': 'Provide at least two doc_ids to compare'}, status_code=400)

    if compare_pipeline is None:
        return JSONResponse({'error': 'Compare pipeline not initialized'}, status_code=500)

    try:
        result = compare_pipeline.compare(doc_ids, topic, top_k=8, max_tokens=max_tokens)
        return JSONResponse(result)
    except Exception as e:
        print(f"Error running compare pipeline: {e}")
        return JSONResponse({'error': str(e)}, status_code=500)


if __name__ == "__main__":
    # Start the FastAPI app with uvicorn for local development
    uvicorn.run(app, host="0.0.0.0", port=8000)
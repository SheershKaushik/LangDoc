
# LangDoc — Retrieval-Augmented Legal/Policy Assistant
Lex6

## Quickstart

Prerequisites:
- Python 3.10+
- A working virtual environment
- (Optional but recommended) Ollama running locally if you want to use the bundled `OllamaLLM`.
- Enough disk space and memory for the embedding model and FAISS index.

Minimal steps (zsh):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# If you already have a persisted vector store, run the app:
python main.py

# If you want to rebuild the index, remove the folder `vector_store_db/` then run:
python main.py
```

Notes:
- On macOS, installing `faiss-cpu` may require a matching wheel or building from source; if you hit issues consider using a Linux environment or a prebuilt wheel for your Python version.
- `OllamaLLM` expects an Ollama server at `http://localhost:11434` by default. Run Ollama and pull a chosen model (e.g. `deepseek-r1:8b`) before attempting queries that require the LLM.

## Project layout (important files)

- `main.py` / `app.py` — CLI driver. Loads or builds the vector store and starts an interactive query loop.
- `requirements.txt`, `pyproject.toml` — dependency manifests.
- `data/` — contains the PDF corpus used for ingestion (many NDC PDF files in this repo).
- `vector_store_db/` — persisted FAISS index and metadata (e.g. `index.faiss`, `docstore.pkl`).

- `src/` — main package:
  - `src/components/data_ingestion.py` — `PDFIngestion`: loads PDFs into LangChain `Document` objects using PyMuPDF.
  - `src/components/data_transformation.py` — `DataTransformation`: splits documents into chunks using `RecursiveCharacterTextSplitter`.
  - `src/components/model_handler.py` — `EmbeddingManager`, `VectorStore`, `FaissDocStore`, `RAGRetriever`, `OllamaLLM`.
  - `src/pipelines/ingestion_pipeline.py` — orchestrates ingest -> split -> embed -> store -> persist.
  - `src/pipelines/prediction_pipeline.py` — orchestrates retrieve -> LLM generate.

## Architecture (data flow)

ASCII diagram:

```
  data/ (PDFs)
	  |
	  v
  PDFIngestion (PyMuPDF)     <-- loads pages -> LangChain Documents
	  |
	  v
  DataTransformation         <-- splits long documents -> chunks
	  |
	  v
  EmbeddingManager          <-- sentence-transformers -> vector embeddings
	  |
	  v
  VectorStore (FAISS)       <-- stores vectors; DocStore holds text & metadata
	  |
	  v
  RAGRetriever              <-- embeds query, FAISS search, filter by score
	  |
	  v
  OllamaLLM (ChatOllama)    <-- generate response given retrieved context
	  |
	  v
  User-facing answer
```

Persistence:
- FAISS index binary is written to `vector_store_db/index.faiss`.
- Document metadata and text mapping is pickled to `vector_store_db/docstore.pkl`.

## How it works (short)
- Ingestion Pipeline: `PDFIngestion` -> `DataTransformation` -> `EmbeddingManager` -> `VectorStore.add_documents()` -> `VectorStore.save_local()`
- Prediction Pipeline: `RAGRetriever.retrieve()` -> build context -> `OllamaLLM.generate_response()` -> return answer




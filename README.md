# Multi-Modal RAG

A retrieval-augmented generation pipeline that ingests PDFs containing text, tables, and images, and answers questions grounded in all three. Document parsing runs through [Unstructured.io](https://unstructured.io/), content summarization and querying run through local [Ollama](https://ollama.com/) models via LangChain, and vectors are stored in [ChromaDB](https://www.trychroma.com/).

## How it works

1. **Partition** — `partition_pdf` (Unstructured, `hi_res` strategy) extracts text, tables (as HTML), and images (as base64) from the source PDF.
2. **Chunk** — `chunk_by_title` groups elements into coherent chunks, capped at 3000 characters.
3. **Summarize** — each chunk is inspected for tables/images. Chunks with mixed content are sent to a local vision-capable Ollama model to generate a rich, searchable text description; the original text/tables/images are preserved in document metadata for the final answer step.
4. **Embed & store** — summarized chunks are embedded with `nomic-embed-text` (via Ollama) and persisted to a local ChromaDB collection, in small batches with retries to tolerate transient local-server hiccups.
5. **Retrieve & answer** — a query is embedded and matched against the store; the top chunks (including their original text, tables, and images) are passed back to the vision model to produce a grounded answer.

The full pipeline is in [multi_modal_rag.ipynb](multi_modal_rag.ipynb).

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (`http://localhost:11434`) with the models used in the notebook pulled, e.g.:
  ```
  ollama pull nomic-embed-text
  ollama pull <vision-capable model used in the notebook>
  ```
- System dependencies for Unstructured's PDF parsing:
  - **Poppler** (PDF text/image extraction)
  - **Tesseract OCR**
  - **libmagic** (file type detection)

  ```bash
  # Linux
  apt-get install poppler-utils tesseract-ocr libmagic-dev

  # macOS
  brew install poppler tesseract libmagic
  ```
  On Windows, install Poppler and Tesseract separately and ensure both are on `PATH`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file for any API keys you need (e.g. if swapping in a hosted LLM/embedding provider).

## Usage

Open [multi_modal_rag.ipynb](multi_modal_rag.ipynb) and run the cells top to bottom, or call the end-to-end pipeline:

```python
db = run_complete_ingestion_pipeline("./docs/your-file.pdf")
```

Then query it:

```python
retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke("your question")
answer = generate_final_answer(chunks, "your question")
```

## Project structure

```
docs/                   Source PDFs
dbv1/, dbv2/             Local ChromaDB persistence directories (generated, gitignored)
chunks_export.json       Exported processed chunks (generated, gitignored)
rag_results.json         Exported retrieval results (generated, gitignored)
multi_modal_rag.ipynb    Main pipeline notebook
requirements.txt         Python dependencies
```

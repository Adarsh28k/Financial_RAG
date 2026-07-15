# Financial RAG over SEC 10-K Filings

This project is a Retrieval-Augmented Generation (RAG) system for answering questions about company 10-K filings. It currently contains Apple's and Google's annual reports, but the pipeline is designed so additional companies can be added with minimal effort.

Everything runs locally:

* **Embedding model:** `all-MiniLM-L12-v2` (stored using Git LFS)
* **Reranker:** Local Cross-Encoder
* **LLM:** Llama 3.1 8B served through Ollama
* **Vector Database:** FAISS

---

# Pipeline

```
PDF
   ↓
Markdown Conversion
   ↓
Semantic Chunking
   ↓
Embeddings + FAISS Index
   ↓
Retrieval + Reranking
   ↓
Llama 3.1 (Ollama)
```

---

# Project Structure

### 1. `pdf_to_markdown.py`

Converts SEC 10-K PDFs into Markdown using **Docling**.

During conversion it also extracts figures and stores them separately.

```
documents/
├── pdf/
├── markdown/
└── images/
```

---

### 2. `chunking.py`

Processes the generated Markdown and creates semantic chunks.

The chunker understands document structure such as:

* Headings
* Paragraphs
* Tables
* Images

Very small fragments (signature blocks, page artifacts, isolated lines, etc.) are merged into nearby chunks so they don't become meaningless retrieval results.

Output:

```
documents/chunks/
```

---

### 3. `embeddings.py`

Creates embeddings for every chunk using the local **all-MiniLM-L12-v2** embedding model.

The embeddings are stored alongside a single FAISS index covering every document.

Output:

```
documents/embeddings/
documents/index/
```

Each chunk also contains metadata (such as the source document), allowing multiple companies to share one index while remaining distinguishable during retrieval.

---

### 4. `retriever.py`

Retrieval happens in two stages.

**Stage 1**

FAISS performs approximate nearest-neighbor search using embedding similarity to retrieve candidate chunks.

**Stage 2**

A Cross-Encoder reranks those candidates by directly evaluating each **(query, passage)** pair.

This significantly improves retrieval quality, especially when semantically similar chunks are not the most relevant answers.

If a question references multiple companies (for example, Apple and Google), retrieval is performed separately for each company before combining the results. This prevents one company's filing from dominating the retrieved context.

---

### 5. `rag_chat.py`

Builds the final prompt from the retrieved context and sends it to a locally hosted **Llama 3.1 8B** model through Ollama.

The model is instructed to answer using only the retrieved context.

---

### `RAG_CHAT.ipynb`

The notebook combines the entire workflow into a single interactive pipeline:

* PDF loading
* Markdown conversion
* Chunking
* Embedding generation
* Index creation
* Retrieval
* Chat interface

This is useful for experimenting without repeatedly restarting models or rerunning individual scripts.

---

# Running the Project

## 1. Clone the repository

```bash
git clone <repository-url>
cd Financial_RAG
```

## 2. Create a virtual environment

```bash
python -m venv .venv

source .venv/bin/activate      # macOS / Linux
# or
.venv\Scripts\activate         # Windows
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Install Ollama

Pull the Llama model:

```bash
ollama pull llama3.1:8b
```

Start Ollama if it isn't already running.

---

## 5. Add documents

Place any SEC 10-K PDF inside:

```
documents/pdf/
```

---

## 6. Build the pipeline

Run the scripts in order:

```text
pdf_to_markdown.py
↓
chunking.py
↓
embeddings.py
↓
retriever.py
```

Or simply execute `RAG_CHAT.ipynb` from top to bottom.

---

## Adding More Companies

Adding another company requires no code changes.

Simply:

1. Place the PDF inside `documents/pdf/`
2. Re-run the pipeline

The new document will be indexed alongside the existing filings while remaining identifiable through its metadata.

---

# Notes

* The current `region` metadata field is only a placeholder and is **not** extracted from the document.
* Chat history is **not** preserved between questions. Each query is answered independently, so follow-up questions should include the necessary context.
* The embedding model is tracked using **Git LFS**, while the Python virtual environment is intentionally **not** included in the repository.
* `discussion.md` contains a detailed write-up of the development process, implementation decisions, and debugging journey.

---

# Tech Stack

* Python
* Docling
* Sentence Transformers
* FAISS
* Cross-Encoder Reranker
* Ollama
* Llama 3.1 8B
* Markdown Processing
* Git LFS

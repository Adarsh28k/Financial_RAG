# RAG over 10-K filings

A small RAG pipeline I built to ask questions over company financial filings (Apple and Google 10-Ks right now, but it's built to take more). Everything runs locally — local embedding model, local cross-encoder reranker, local Llama 3.1 8B through Ollama for the actual answers.

## How it works

```
PDF → Markdown → Chunks → Embeddings + FAISS Index → Retrieval → Chat
```

1. **`pdf_to_markdown.py`** — converts PDFs (`documents/pdf/`) into markdown using docling, pulling out figures into `documents/images/`.
2. **`chunking.py`** — parses the markdown into headings/paragraphs/tables/images and chunks it, merging short fragments (signature blocks, stray lines) so they don't end up as useless standalone chunks. Output goes to `documents/chunks/`.
3. **`embeddings.py`** — embeds every chunk with a local `all-MiniLM-L12-v2` model and builds one FAISS index across all documents. Output goes to `documents/embeddings/` and `documents/index/`.
4. **`retreiver.py`** — retrieval is two-stage: cosine similarity narrows things down, then a cross-encoder reranks the survivors by actually scoring the (query, passage) pair, which catches relevant chunks a plain embedding comparison misses.
5. **`rag_chat.py`** — builds a prompt from whatever got retrieved and asks a local Llama 3.1 8B (via Ollama) to answer using only that context. If a question mentions more than one company, it retrieves each one separately first so one company can't crowd the other out of the results.

`RAG_CHAT.ipynb` ties all of this together in one notebook, from PDF loading through an actual chat loop at the end — good for playing with the pipeline interactively without waiting on model load times every run.

## Running it

Everything lives in one environment now (`venv311` — see `requirements.txt`), plus Ollama installed separately with `llama3.1:8b` pulled. Drop a PDF into `documents/pdf/`, run the four scripts in order (or just run the notebook top to bottom), and query away.

Adding another company is just: drop its PDF in, rerun the pipeline. Everything after that stays in one shared index, distinguished by a `document` field in each chunk's metadata.

## Worth knowing

- A chunk's `region` metadata field isn't real — it's a placeholder, not derived from the document. Ignore it for now.
- The chat cell has no memory across turns — every question is answered fresh, so keep questions self-contained rather than referring back to earlier ones.
- `discussion.md` has the full, much longer story of everything that broke along the way and how I chased each thing down, if you're into that kind of read.

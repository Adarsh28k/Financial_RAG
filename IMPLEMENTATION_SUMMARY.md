# RAG Pipeline Implementation Summary

This document summarizes the implementation logic for the current Python modules created so far in this project.

## 1. `pdf_to_markdown.py`

### Responsibility
Convert all PDF files in a folder into Markdown files, extracting text, tables, and images.

### Key logic
- Imports `DocumentConverter` from `docling.document_converter`.
- Uses `Pillow` to save extracted images as PNG files.
- For each PDF in the input folder:
  - Convert the PDF using Docling.
  - Traverse the returned document items.
  - Emit headings as Markdown headings (`#`, `##`, `###`, `####`).
  - Emit paragraphs directly as Markdown text.
  - Convert tables into Markdown pipe-table format.
  - Extract images and save them in the same folder, then add relative Markdown image links.
- Saves a `.md` file for each `.pdf` using the same base name.

### Main functions
- `extract_pdf_to_markdown(pdf_path: str) -> tuple[str, int]`
  - Converts one PDF to Markdown and returns content with extracted image count.
- `convert_table_to_markdown(table) -> str`
  - Builds a Markdown table string from a Docling table object.
- `process_folder(folder_path: str)`
  - Processes all PDFs in a folder and writes output markdown files.

## 2. `chunking.py`

### Responsibility
Read generated Markdown files, preserve structural blocks, and create chunks for retrieval.

### Key logic
- Parses Markdown into block types: headings, paragraphs, tables, images.
- Keeps tables and images intact in chunks.
- Includes one paragraph before and after every table/image when available.
- Uses LangChain text splitter if installed; otherwise uses a simple character-based fallback.
- Generates metadata for each chunk:
  - `document`
  - `page` (currently `None`)
  - `heading`
  - `subheading`
  - `region` (cycled from a hardcoded country list)
  - `chunk_index`
- Computes chunk statistics:
  - total chunks
  - min/max/average chunk size
  - average words per chunk
  - size distribution buckets
- Writes chunks to a JSONL file per Markdown document.

### Main functions
- `simple_split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]`
  - Fallback text splitter if LangChain is unavailable.
- `parse_markdown_blocks(md: str) -> List[Dict[str, Any]]`
  - Converts Markdown content into a list of typed blocks.
- `chunk_blocks(blocks: List[Dict[str, Any]], doc_name: str) -> List[Dict[str, Any]]`
  - Builds chunk objects from parsed blocks with metadata.
- `compute_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]`
  - Calculates chunk statistics.
- `process_folder(folder_path: str)`
  - Reads `.md` files, creates chunks, prints stats, and saves `_chunks.jsonl` files.

## 3. `embeddings.py`

### Responsibility
Load chunk JSONL data, encode chunk texts to embeddings, and persist a FAISS index.

### Key logic
- Loads all `*_chunks.jsonl` files from the folder.
- Uses `sentence-transformers/all-mpnet-base-v2` to create embeddings.
- Normalizes embeddings with L2 normalization.
- Builds a FAISS `IndexFlatIP` index wrapped with `IndexIDMap`.
- Persists:
  - `faiss_index.idx`
  - `chunk_embeddings.jsonl`

### Main functions
- `load_chunked_data(folder_path: str) -> List[Dict[str, Any]]`
  - Reads chunk metadata and text from JSONL files.
- `create_embeddings(texts: List[str], model_name: str) -> np.ndarray`
  - Generates normalized embeddings from text chunks.
- `build_faiss_index(embeddings: np.ndarray) -> faiss.Index`
  - Builds the FAISS index using inner product for cosine similarity.
- `save_index(index: faiss.Index, path: Path) -> None`
  - Writes the FAISS index to disk.
- `save_metadata(chunks: List[Dict[str, Any]], path: Path) -> None`
  - Persists chunk metadata as JSONL.
- `process_folder(folder_path: str) -> None`
  - Orchestrates loading, encoding, index creation, and persistence.

## 4. `retrieval.py`

### Responsibility
Load persisted FAISS index and metadata, apply filters, search by query, and return top chunks.

### Key logic
- Loads `faiss_index.idx` and `chunk_embeddings.jsonl`.
- Parses metadata filters from `key=value` pairs.
- Filters candidate chunks by metadata before vector search.
- Optionally filters by heading or subheading.
- Uses `sentence-transformers/all-mpnet-base-v2` to embed the query.
- Computes cosine similarity and drops chunks below `0.60`.
- Reranks results using a mix of cosine similarity and lexical overlap.
- Returns top 3 chunks with scores and metadata.

### Main functions
- `load_index(folder: Path) -> faiss.Index`
  - Loads the persisted FAISS index.
- `load_metadata(folder: Path) -> List[Dict[str, Any]]`
  - Loads chunk metadata from JSONL.
- `parse_metadata_filter(filter_str: Optional[str]) -> Dict[str, str]`
  - Parses CLI metadata filter strings.
- `apply_metadata_filters(records: List[Dict[str, Any]], filters: Dict[str, str]) -> List[int]`
  - Returns indices that match metadata filters.
- `embed_query(model: SentenceTransformer, text: str) -> np.ndarray`
  - Encodes and normalizes the query vector.
- `compute_similarities_by_ids(index: faiss.Index, query_vec: np.ndarray, ids: List[int]) -> List[Tuple[int, float]]`
  - Computes similarity for candidate IDs.
- `lexical_overlap_score(query: str, text: str) -> float`
  - Computes a lightweight lexical similarity score.
- `rerank(results: List[Tuple[int, float]], records: List[Dict[str, Any]], query: str) -> List[Tuple[int, float, float]]`
  - Reranks using cosine + lexical overlap.
- `retrieve(folder: Path, query: str, meta_filter: Dict[str, str], heading: Optional[str], subheading: Optional[str]) -> Dict[str, Any]`
  - Produces the final retrieval result.
- `main()`
  - CLI entrypoint for search.

## 5. `rag_chat.py`

### Responsibility
Build a full RAG answer generator that uses retrieval results to prompt an LLM and return a final answer.

### Key logic
- Imports `retrieve()` from `retrieval.py`.
- Uses the top retrieval results as the source context.
- Builds a prompt with query, chunk excerpts, metadata, and explicit instructions.
- Calls Ollama with `llama3.3` by default.
- Returns a JSON object containing:
  - the original query
  - retrieval results
  - the LLM answer
  - model and temperature metadata

### Main functions
- `parse_metadata_filter(filter_str: Optional[str]) -> Dict[str, str]`
  - Parses CLI metadata filter strings.
- `build_prompt(query: str, results: List[Dict[str, Any]]) -> str`
  - Constructs a prompt containing retrieved chunks and answer instructions.
- `call_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> str`
  - Sends the prompt to Ollama via CLI and returns the generated text.
- `answer_query(folder: Path, query: str, meta_filter: Optional[str], heading: Optional[str], subheading: Optional[str], model: str, temperature: float) -> Dict[str, Any]`
  - Runs retrieval and the final LLM answer generation.
- `main()`
  - CLI entrypoint for the chat layer.

## Notes
- The current implementation stores extracted Markdown and chunk metadata with relative paths for images.
- The pipeline is designed to be incremental:
  1. `pdf_to_markdown.py` produces Markdown and image files.
  2. `chunking.py` converts Markdown into structured chunks.
  3. `embeddings.py` creates embeddings and a FAISS index.
  4. `retrieval.py` performs filtered vector retrieval.

## Files created so far
- `pdf_to_markdown.py`
- `chunking.py`
- `embeddings.py`
- `retrieval.py`
- `rag_chat.py`
- `IMPLEMENTATION_SUMMARY.md`

## How to use the files
- Run `python pdf_to_markdown.py <folder>` first to generate Markdown.
- Run `python chunking.py <folder>` to generate chunk JSONL.
- Run `python embeddings.py <folder>` to generate embeddings and index.
- Run `python retrieval.py <folder> "query" [--meta key=value] [--heading "..."] [--subheading "..."]` to retrieve top chunks.
- Run `python rag_chat.py <folder> "query" [--meta key=value] [--heading "..."] [--subheading "..."]` to get a final LLM answer over retrieved chunks.

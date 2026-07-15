import os

# keep every math lib on one thread - avoids the semaphore/thread conflicts
# that torch + faiss + tokenizers tend to trigger on macOS/conda
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(1)

CHUNKS_SUBDIR = "chunks"
INDEX_SUBDIR = "index"
EMBEDDINGS_SUBDIR = "embeddings"
INDEX_FILENAME = "faiss_index.idx"
METADATA_FILENAME = "chunk_embeddings.jsonl"
MODEL_NAME = "/Users/sarthakkumar/Desktop/RAG/all-MiniLM-L12-v2"


def load_chunked_data(folder_path: str) -> list[dict]:
    chunks_dir = Path(folder_path) / CHUNKS_SUBDIR
    chunk_files = sorted(chunks_dir.glob("*_chunks.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk JSONL files found in {chunks_dir}. Run chunking.py first.")

    chunks = []
    for chunk_file in chunk_files:
        with chunk_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def create_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name, device="cpu")
    print("Model loaded. Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=8, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
    ids = np.arange(embeddings.shape[0], dtype=np.int64)
    index.add_with_ids(embeddings, ids)
    return index


def save_metadata(chunks: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def process_folder(folder_path: str) -> None:
    folder = Path(folder_path)
    index_dir = folder / INDEX_SUBDIR
    embeddings_dir = folder / EMBEDDINGS_SUBDIR
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunked_data(folder_path)
    texts = [chunk["text"] for chunk in chunks]
    print(f"Loaded {len(chunks)} chunks from {folder / CHUNKS_SUBDIR}")

    embeddings = create_embeddings(texts, MODEL_NAME)
    print(f"Generated embeddings shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(index_dir / INDEX_FILENAME))
    print(f"Saved FAISS index to {index_dir / INDEX_FILENAME}")

    save_metadata(chunks, embeddings_dir / METADATA_FILENAME)
    print(f"Saved chunk metadata to {embeddings_dir / METADATA_FILENAME}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python embeddings.py <folder_path>")
        sys.exit(1)

    process_folder(sys.argv[1])
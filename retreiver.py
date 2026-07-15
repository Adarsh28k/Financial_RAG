import os

# Loading faiss + torch (via sentence-transformers) together in one process can
# crash with duplicate/conflicting OpenMP runtimes on macOS - these three env
# vars have to be set before either library is imported.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

MODEL = SentenceTransformer("/Users/sarthakkumar/Desktop/RAG/all-MiniLM-L12-v2", device="cpu")
CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

INDEX_SUBDIR = "index"
EMBEDDINGS_SUBDIR = "embeddings"
INDEX_FILENAME = "faiss_index.idx"
METADATA_FILENAME = "chunk_embeddings.jsonl"
COSINE_THRESHOLD = 0.30
CROSS_ENCODER_TOP_K = 50  # candidates (post cosine filter) sent to the cross-encoder


def load_index(folder: Path) -> faiss.Index:
    return faiss.read_index(str(folder / INDEX_SUBDIR / INDEX_FILENAME))


def load_metadata(folder: Path) -> List[Dict[str, Any]]:
    meta_path = folder / EMBEDDINGS_SUBDIR / METADATA_FILENAME
    with meta_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_metadata_filter(filter_str: Optional[str]) -> Dict[str, str]:
    if not filter_str:
        return {}
    out = {}
    for part in filter_str.split(','):
        if '=' in part:
            key, value = part.split('=', 1)
            out[key.strip()] = value.strip()
    return out


def apply_metadata_filters(records: List[Dict[str, Any]], filters: Dict[str, str]) -> List[int]:
    if not filters:
        return list(range(len(records)))
    return [
        i for i, rec in enumerate(records)
        if all(str(rec.get('metadata', {}).get(k, '')).lower() == v.lower() for k, v in filters.items())
    ]


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode([text], convert_to_numpy=True).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    return vec


def compute_similarities_by_ids(index: faiss.Index, query_vec: np.ndarray, ids: List[int]) -> List[Tuple[int, float]]:
    # IDs are assigned sequentially at build time (see embeddings.py), so they line
    # up directly with positions in the underlying flat index - reconstructing by
    # id and taking a dot product with the (already normalized) query is an exact
    # cosine similarity, no approximate search needed.
    query_flat = query_vec.ravel()
    sims = []
    for _id in ids:
        vec = index.index.reconstruct(_id)
        sims.append((_id, float(np.dot(query_flat, vec))))
    return sims


def rerank(results: List[Tuple[int, float]], records: List[Dict[str, Any]], query: str) -> List[Tuple[int, float, float]]:
    # Stage 1: cosine gate + cap, just to bound how many pairs the cross-encoder scores.
    candidates = [(_id, cosine) for _id, cosine in results if cosine >= COSINE_THRESHOLD]
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:CROSS_ENCODER_TOP_K]

    if not candidates:
        return []

    # Stage 2: cross-encoder scores the actual (query, passage) pair jointly, so it
    # catches relevance cosine misses (e.g. a term buried in a longer chunk).
    pairs = [(query, records[_id]["text"]) for _id, _ in candidates]
    raw_scores = CROSS_ENCODER.predict(pairs)
    cross_scores = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=np.float64)))

    scored = [
        (_id, cosine, float(cross_score))
        for (_id, cosine), cross_score in zip(candidates, cross_scores)
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def retrieve(
    folder: Path,
    query: str,
    meta_filter: Dict[str, str],
    heading: Optional[str] = None,
    subheading: Optional[str] = None,
) -> Dict[str, Any]:
    index = load_index(folder)
    records = load_metadata(folder)

    candidate_ids = apply_metadata_filters(records, meta_filter)

    if heading:
        candidate_ids = [
            i for i in candidate_ids
            if (records[i].get("metadata", {}).get("heading") or "").lower() == heading.lower()
        ]

    if subheading:
        candidate_ids = [
            i for i in candidate_ids
            if (records[i].get("metadata", {}).get("subheading") or "").lower() == subheading.lower()
        ]

    qvec = embed_query(MODEL, query)
    results = compute_similarities_by_ids(index, qvec, candidate_ids)
    reranked = rerank(results, records, query)

    return {
        "query": query,
        "results": [
            {
                "id": idx,
                "cosine": cosine,
                "final_score": final,
                "text": records[idx]["text"],
                "metadata": records[idx].get("metadata", {}),
            }
            for idx, cosine, final in reranked[:3]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Folder with FAISS index and metadata')
    parser.add_argument('query', help='User query')
    parser.add_argument('--meta', help='Metadata filters, comma-separated key=value pairs', default=None)
    parser.add_argument('--heading', help='Heading filter', default=None)
    parser.add_argument('--subheading', help='Subheading filter', default=None)
    args = parser.parse_args()

    result = retrieve(
        Path(args.folder),
        args.query,
        parse_metadata_filter(args.meta),
        args.heading,
        args.subheading,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from RAG.retreiver import retrieve, load_metadata

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
RESULTS_PER_DOCUMENT = 3
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def parse_metadata_filter(filter_str: Optional[str]) -> Dict[str, str]:
    if not filter_str:
        return {}
    out: Dict[str, str] = {}
    for part in filter_str.split(','):
        if '=' in part:
            key, value = part.split('=', 1)
            out[key.strip()] = value.strip()
    return out


def detect_documents(query: str, documents: List[str]) -> List[str]:
    """Match a query against known `document` metadata values via a simple alias
    (the part of the document name before its first underscore, e.g.
    GOOGLE_FINANCE -> "google"). Returns every document mentioned in the query."""
    query_lower = query.lower()
    matched = []
    for doc in documents:
        alias = doc.split('_')[0].lower()
        if alias and alias in query_lower:
            matched.append(doc)
    return matched


def retrieve_multi(
    folder: Path,
    query: str,
    meta_filter: Dict[str, str],
    heading: Optional[str],
    subheading: Optional[str],
) -> Dict[str, Any]:
    # Explicit filters always win over auto-detected ones.
    if meta_filter:
        return retrieve(folder, query, meta_filter, heading, subheading)

    records = load_metadata(folder)
    documents = sorted({
        r.get('metadata', {}).get('document')
        for r in records
        if r.get('metadata', {}).get('document')
    })
    matched = detect_documents(query, documents)

    if len(matched) <= 1:
        filt = {'document': matched[0]} if matched else {}
        return retrieve(folder, query, filt, heading, subheading)

    # Query spans multiple documents - retrieve each separately so one document's
    # higher-scoring chunks can't crowd another out of a single global top-k.
    combined_results = []
    for doc in matched:
        result = retrieve(folder, query, {'document': doc}, heading, subheading)
        combined_results.extend(result.get('results', [])[:RESULTS_PER_DOCUMENT])

    return {
        'query': query,
        'matched_documents': matched,
        'results': combined_results,
    }


def build_prompt(query: str, results: List[Dict[str, Any]]) -> str:
    header = [
        "You are a helpful assistant that answers questions using only the provided document excerpts.",
        "If the answer cannot be found in the provided sources, say explicitly that you don't know based on the given documents.",
        "Do not invent facts or source content that is not present in the excerpts.",
        "",
        f"Question: {query}",
        "",
        "Sources:",
    ]

    if not results:
        header.append("No document chunks were retrieved for this query.")
    else:
        for idx, item in enumerate(results, start=1):
            metadata = item.get('metadata', {})
            heading = metadata.get('heading') or 'N/A'
            subheading = metadata.get('subheading') or 'N/A'
            document = metadata.get('document') or 'unknown'
            region = metadata.get('region') or 'unknown'
            header.append(f"Source {idx} | document={document} | heading={heading} | subheading={subheading} | region={region} | score={item.get('cosine', 0.0):.3f}")
            header.append(item.get('text', '').strip())
            header.append("")

    header.append("Answer the question using only the above excerpts.")
    header.append("If you are unsure, say 'I don't know based on the given documents.'")
    return "\n".join(header)


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> str:
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def answer_query(
    folder: Path,
    query: str,
    meta_filter: Optional[str] = None,
    heading: Optional[str] = None,
    subheading: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    filters = parse_metadata_filter(meta_filter)
    retrieval_result = retrieve_multi(folder, query, filters, heading, subheading)
    prompt = build_prompt(query, retrieval_result.get('results', []))
    llm_answer = call_ollama(prompt, model=model, temperature=temperature)

    return {
        'query': query,
        'model': model,
        'temperature': temperature,
        'retrieval': retrieval_result,
        'answer': llm_answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run RAG question answering over retrieved PDF chunks.')
    parser.add_argument('folder', help='Folder containing FAISS index and metadata files')
    parser.add_argument('query', help='User query to answer')
    parser.add_argument('--meta', help='Metadata filter as comma-separated key=value pairs', default=None)
    parser.add_argument('--heading', help='Filter by heading', default=None)
    parser.add_argument('--subheading', help='Filter by subheading', default=None)
    parser.add_argument('--model', help='Ollama model name', default=DEFAULT_MODEL)
    parser.add_argument('--temperature', help='Ollama temperature', type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    folder = Path(args.folder)
    response = answer_query(
        folder=folder,
        query=args.query,
        meta_filter=args.meta,
        heading=args.heading,
        subheading=args.subheading,
        model=args.model,
        temperature=args.temperature,
        
    )

    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

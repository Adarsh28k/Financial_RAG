import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

# Try to import LangChain text splitter; fallback to a simple splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None


MARKDOWN_SUBDIR = "markdown"
CHUNKS_SUBDIR = "chunks"

COUNTRIES = [
    "United States", "Canada", "United Kingdom", "Germany", "France",
    "Spain", "Italy", "Netherlands", "Sweden", "Norway",
    "Denmark", "Australia", "New Zealand", "Japan", "South Korea",
    "India", "Brazil", "Mexico", "South Africa", "Nigeria",
]


def simple_split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = max(end - chunk_overlap, end)
        if start == end:
            start = end
    return chunks


def merge_short_paragraphs(texts: List[str], min_chars: int) -> List[str]:
    """Merge consecutive short paragraphs (e.g. signature-block lines) so they
    don't end up as standalone, near-empty chunks with no surrounding context."""
    merged = []
    buffer = ""
    for t in texts:
        buffer = f"{buffer}\n\n{t}" if buffer else t
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{buffer}"
        else:
            merged.append(buffer)
    return merged


def split_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    return simple_split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def parse_markdown_blocks(md: str) -> List[Dict[str, Any]]:
    lines = md.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Heading
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            blocks.append({"type": "heading", "level": level, "text": text})
            i += 1
            continue

        # Image
        m = re.match(r'^!\[[^\]]*\]\(([^)]+)\)', line)
        if m:
            img_md = line.strip()
            blocks.append({"type": "image", "text": img_md})
            i += 1
            continue

        # Table: lines starting with | or with pipe-separated rows
        if line.strip().startswith('|'):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                tbl_lines.append(lines[i])
                i += 1
            blocks.append({"type": "table", "text": "\n".join(tbl_lines)})
            continue

        # Blank line
        if line.strip() == "":
            i += 1
            continue

        # Paragraph: collect until blank line or block start
        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() != "" and not re.match(r'^(#{1,6})\s+', lines[i]) and not lines[i].strip().startswith('|') and not re.match(r'^!\[[^\]]*\]\(([^)]+)\)', lines[i]):
            para_lines.append(lines[i])
            i += 1
        para_text = "\n".join(para_lines).strip()
        blocks.append({"type": "paragraph", "text": para_text})
    return blocks


def chunk_blocks(blocks: List[Dict[str, Any]], doc_name: str, min_chunk_chars: int = 120) -> List[Dict[str, Any]]:
    # Phase 1: figure out which paragraphs are absorbed as neighbors of a table/image,
    # and precompute the resulting table/image chunk text keyed by block index.
    used_para_indices = set()
    table_image_text = {}
    for idx, blk in enumerate(blocks):
        if blk["type"] not in ("table", "image"):
            continue
        parts = []
        if idx - 1 >= 0 and blocks[idx - 1]["type"] == "paragraph":
            parts.append(blocks[idx - 1]["text"])
            used_para_indices.add(idx - 1)
        parts.append(blk["text"])
        if idx + 1 < len(blocks) and blocks[idx + 1]["type"] == "paragraph":
            parts.append(blocks[idx + 1]["text"])
            used_para_indices.add(idx + 1)
        chunk_text = "\n\n".join(parts).strip()
        if chunk_text:
            table_image_text[idx] = chunk_text

    # Phase 2: single forward pass over blocks, in document order, so chunk order
    # always matches source order (no separate interleaving guesswork needed).
    final_chunks = []
    current_h1 = None
    current_h2 = None
    i = 0
    n = len(blocks)

    while i < n:
        blk = blocks[i]

        if blk["type"] == "heading":
            if blk["level"] == 1:
                current_h1 = blk["text"]
                current_h2 = None
            elif blk["level"] == 2:
                current_h2 = blk["text"]
            i += 1
            continue

        if blk["type"] in ("table", "image"):
            chunk_text = table_image_text.get(i)
            if chunk_text:
                final_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "document": doc_name,
                        "page": None,
                        "heading": current_h1,
                        "subheading": current_h2,
                    },
                })
            i += 1
            continue

        if blk["type"] == "paragraph":
            if i in used_para_indices:
                i += 1
                continue

            # Collect the run of consecutive, not-yet-used paragraphs under this
            # heading so short ones (e.g. signature-block lines) can be merged
            # with their neighbors instead of becoming standalone chunks.
            run_texts = []
            while i < n and blocks[i]["type"] == "paragraph" and i not in used_para_indices:
                run_texts.append(blocks[i]["text"])
                i += 1

            for merged_text in merge_short_paragraphs(run_texts, min_chunk_chars):
                for sub in split_text(merged_text):
                    final_chunks.append({
                        "text": sub.strip(),
                        "metadata": {
                            "document": doc_name,
                            "page": None,
                            "heading": current_h1,
                            "subheading": current_h2,
                        },
                    })
            continue

        i += 1

    # Assign regions (cycle through COUNTRIES)
    for i, c in enumerate(final_chunks):
        region = COUNTRIES[i % len(COUNTRIES)]
        c.setdefault("metadata", {})["region"] = region
        c.setdefault("metadata", {})["chunk_index"] = i
    return final_chunks


def compute_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    sizes = [len(c["text"]) for c in chunks]
    words = [len(c["text"].split()) for c in chunks]
    if not sizes:
        return {}
    stats = {
        "total_chunks": len(chunks),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": int(statistics.mean(sizes)),
        "avg_words_per_chunk": float(statistics.mean(words)),
        "size_distribution": {
            "<500": sum(1 for s in sizes if s < 500),
            "500-1000": sum(1 for s in sizes if 500 <= s < 1000),
            "1000-2000": sum(1 for s in sizes if 1000 <= s < 2000),
            ">=2000": sum(1 for s in sizes if s >= 2000),
        },
    }
    return stats


def process_folder(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    markdown_dir = folder / MARKDOWN_SUBDIR
    chunks_dir = folder / CHUNKS_SUBDIR
    chunks_dir.mkdir(parents=True, exist_ok=True)

    md_files = list(markdown_dir.glob("*.md"))
    if not md_files:
        print(f"No Markdown files found in '{markdown_dir}'")
        return

    for md_file in md_files:
        print(f"Processing: {md_file.name}")
        md_text = md_file.read_text(encoding="utf-8")
        blocks = parse_markdown_blocks(md_text)
        doc_name = md_file.stem
        chunks = chunk_blocks(blocks, doc_name)

        stats = compute_stats(chunks)
        print("Chunk stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Save chunks as JSONL
        out_path = chunks_dir / (md_file.stem + "_chunks.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        print(f"Saved {len(chunks)} chunks to {out_path.name}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chunking.py <folder_path>")
        sys.exit(1)
    process_folder(sys.argv[1])

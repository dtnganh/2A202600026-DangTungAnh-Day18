"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE, SEMANTIC_THRESHOLD


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load non-empty markdown/text files from data/."""
    docs = []
    patterns = ["*.md", "*.txt"]
    for pattern in patterns:
        for fp in sorted(glob.glob(os.path.join(data_dir, pattern))):
            with open(fp, encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Basic paragraph chunking baseline."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group adjacent sentences by lexical similarity as an offline semantic fallback."""
    metadata = metadata or {}
    sentences = _sentences(text)
    if not sentences:
        return []

    effective_threshold = min(threshold, 0.35)
    groups: list[list[str]] = [[sentences[0]]]
    for sentence in sentences[1:]:
        previous = groups[-1][-1]
        same_header_area = sentence.startswith("#") or previous.startswith("#")
        if not same_header_area and _jaccard(previous, sentence) < effective_threshold:
            groups.append([])
        groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "chunk_index": i, "strategy": "semantic"},
        )
        for i, group in enumerate(groups)
        if " ".join(group).strip()
    ]


def _paragraph_groups(text: str, target_size: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    groups = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target_size:
            groups.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        groups.append(current.strip())
    return groups


def _sliding_windows(text: str, size: int, overlap: int = 40) -> list[str]:
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    windows = []
    step = max(size - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            windows.append(piece)
        if start + size >= len(text):
            break
    return windows


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks for context and child chunks for precise retrieval."""
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(_paragraph_groups(text, parent_size)):
        pid = f"{metadata.get('source', 'doc')}_parent_{parent_index}"
        parents.append(
            Chunk(
                text=parent_text,
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": parent_index},
            )
        )
        for child_index, child_text in enumerate(_sliding_windows(parent_text, child_size)):
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                        "chunk_index": child_index,
                        "parent_id": pid,
                    },
                    parent_id=pid,
                )
            )
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Parse Markdown headers and keep each section together."""
    metadata = metadata or {}
    sections = re.split(r"(^#{1,6}\s+.+$)", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    current_header = metadata.get("source", "Document")
    current_content = ""

    for part in sections:
        if not part:
            continue
        if re.match(r"^#{1,6}\s+", part):
            if current_content.strip():
                chunks.append(
                    Chunk(
                        text=f"{current_header}\n{current_content}".strip(),
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                            "section": current_header.lstrip("# ").strip(),
                            "strategy": "structure",
                        },
                    )
                )
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    if current_content.strip():
        chunks.append(
            Chunk(
                text=f"{current_header}\n{current_content}".strip(),
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "section": current_header.lstrip("# ").strip(),
                    "strategy": "structure",
                },
            )
        )

    return chunks or chunk_basic(text, metadata={**metadata, "strategy": "structure"})


def _stats(chunks: list[Chunk]) -> dict:
    lengths = [len(c.text) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies on documents and return compact stats."""
    buckets = {"basic": [], "semantic": [], "hierarchical": [], "structure": []}

    for doc in documents:
        text, metadata = doc["text"], doc.get("metadata", {})
        buckets["basic"].extend(chunk_basic(text, metadata=metadata))
        buckets["semantic"].extend(chunk_semantic(text, metadata=metadata))
        _, children = chunk_hierarchical(text, metadata=metadata)
        buckets["hierarchical"].extend(children)
        buckets["structure"].extend(chunk_structure_aware(text, metadata=metadata))

    results = {name: _stats(chunks) for name, chunks in buckets.items()}
    print(f"{'Strategy':<14} | {'Chunks':>6} | {'Avg Len':>7} | {'Min':>5} | {'Max':>5}")
    for name, stats in results.items():
        print(
            f"{name:<14} | {stats['num_chunks']:>6} | {stats['avg_length']:>7} | "
            f"{stats['min_length']:>5} | {stats['max_length']:>5}"
        )
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")

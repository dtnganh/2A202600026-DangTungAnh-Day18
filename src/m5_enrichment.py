"""
Module 5: Enrichment Pipeline
==============================
OpenAI API enrichment for chunks before indexing.
"""

import os
import hashlib
import json
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, OPENAI_API_KEY, OPENAI_ENRICH_BATCH_SIZE, OPENAI_MODEL, USE_OPENAI_ENRICHMENT
from src.progress import ProgressBar


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for M5 enrichment.")
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)


def _chat(messages: list[dict], max_tokens: int = 300, temperature: float = 0.1) -> str:
    response = _openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [part.strip() for part in parts if part.strip()]


def summarize_chunk(text: str) -> str:
    """Create a short summary with OpenAI."""
    return _chat(
        [
            {
                "role": "system",
                "content": "Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn bằng tiếng Việt. Chỉ trả về phần tóm tắt.",
            },
            {"role": "user", "content": text[:4000]},
        ],
        max_tokens=140,
    )


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions a chunk may answer using OpenAI."""
    live = _chat(
        [
            {
                "role": "system",
                "content": (
                    f"Tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
                    "Mỗi câu hỏi trên một dòng, không đánh số."
                ),
            },
            {"role": "user", "content": text[:4000]},
        ],
        max_tokens=220,
    )
    questions = [line.strip().lstrip("-0123456789. )") for line in live.splitlines() if line.strip()]
    return questions[:n_questions]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a short source-aware context sentence while preserving original text."""
    live = _chat(
        [
            {
                "role": "system",
                "content": (
                    "Viết đúng 1 câu ngắn bằng tiếng Việt mô tả đoạn văn nằm trong tài liệu nào "
                    "và nói về chủ đề gì. Không thêm giải thích."
                ),
            },
            {"role": "user", "content": f"Tài liệu: {document_title or 'tài liệu nguồn'}\n\nĐoạn văn:\n{text[:3500]}"},
        ],
        max_tokens=80,
    )
    return f"{live}\n\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract metadata with OpenAI."""
    live = _chat(
        [
            {
                "role": "system",
                "content": (
                    'Trích xuất metadata từ đoạn văn. Chỉ trả về JSON hợp lệ với schema: '
                    '{"topic": "string", "entities": ["string"], "category": "policy|finance|general", "language": "vi|en"}.'
                ),
            },
            {"role": "user", "content": text[:4000]},
        ],
        max_tokens=180,
    )
    start = live.find("{")
    end = live.rfind("}") + 1
    parsed = json.loads(live[start:end])
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI metadata response was not a JSON object.")
    parsed.setdefault("entities", [])
    parsed.setdefault("language", "vi")
    return parsed


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Run selected enrichment methods on chunks."""
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for enrich_chunks().")
    if not USE_OPENAI_ENRICHMENT:
        raise RuntimeError("USE_OPENAI_ENRICHMENT must be 1 when offline fallback is disabled.")

    return _enrich_batches_with_openai(chunks, methods)


def _enrich_batches_with_openai(chunks: list[dict], methods: list[str]) -> list[EnrichedChunk]:
    """Enrich all chunks through OpenAI in batches to keep API calls practical."""
    enriched: list[EnrichedChunk] = []
    total_batches = (len(chunks) + OPENAI_ENRICH_BATCH_SIZE - 1) // OPENAI_ENRICH_BATCH_SIZE
    progress = ProgressBar("  M5 OpenAI enrichment", total_batches)
    for start in range(0, len(chunks), OPENAI_ENRICH_BATCH_SIZE):
        batch = chunks[start : start + OPENAI_ENRICH_BATCH_SIZE]
        batch_payload = [
            {
                "index": start + offset,
                "source": chunk.get("metadata", {}).get("source", ""),
                "text": chunk.get("text", "")[:1200],
            }
            for offset, chunk in enumerate(batch)
        ]
        batch_no = start // OPENAI_ENRICH_BATCH_SIZE + 1
        cache_path = _batch_cache_path(batch_payload, methods)
        if os.path.exists(cache_path):
            progress.update(batch_no - 1, f"batch {batch_no}/{total_batches} cached")
            with open(cache_path, encoding="utf-8") as f:
                response_items = json.load(f)
        else:
            progress.update(batch_no - 1, f"batch {batch_no}/{total_batches} calling API")
            response_items = _enrich_batch_with_openai_with_retry(batch_payload, methods, expected_count=len(batch))
            _write_batch_cache(cache_path, response_items)
        by_index = {int(item.get("index", -1)): item for item in response_items if "index" in item}
        for offset, chunk in enumerate(batch):
            global_index = start + offset
            item = by_index.get(global_index)
            if item is None and len(response_items) == len(batch):
                item = response_items[offset]
            if item is None:
                raise ValueError(f"OpenAI enrichment missing chunk index {global_index}")
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            context = item.get("context", "").strip()
            enriched_text = f"{context}\n\n{text}" if "contextual" in methods or "full" in methods else text
            enriched.append(
                EnrichedChunk(
                    original_text=text,
                    enriched_text=enriched_text,
                    summary=item.get("summary", ""),
                    hypothesis_questions=item.get("hypothesis_questions", [])[:3],
                    auto_metadata={**metadata, **item.get("metadata", {})},
                    method="+".join(methods),
                )
            )
        progress.update(batch_no, f"batch {batch_no}/{total_batches} done")
    return enriched


def _batch_cache_path(batch_payload: list[dict], methods: list[str]) -> str:
    """Return a cache file path for API enrichment output."""
    cache_root = os.path.join(DATA_DIR, ".openai_enrich_cache")
    payload = {
        "version": 1,
        "model": OPENAI_MODEL,
        "methods": methods,
        "chunks": batch_payload,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return os.path.join(cache_root, f"{digest}.json")


def _write_batch_cache(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _enrich_batch_with_openai_with_retry(batch_payload: list[dict], methods: list[str], expected_count: int) -> list[dict]:
    for attempt in range(1, 4):
        items = _enrich_batch_with_openai(batch_payload, methods)
        if len(items) == expected_count:
            return items
        print(f"  OpenAI enrichment returned {len(items)}/{expected_count}; retrying batch (attempt {attempt}/3)")
    raise ValueError("OpenAI enrichment did not return enough items after retries.")


def _enrich_batch_with_openai(batch_payload: list[dict], methods: list[str]) -> list[dict]:
    client = _openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        max_tokens=8000,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn làm enrichment cho RAG tiếng Việt. Chỉ trả về JSON array hợp lệ. "
                    "Mỗi phần tử giữ nguyên index input và có schema: "
                    '{"index": number, "summary": "string", "context": "string", '
                    '"hypothesis_questions": ["string"], '
                    '"metadata": {"topic": "string", "entities": ["string"], '
                    '"category": "policy|finance|general", "language": "vi|en"}}. '
                    "context đúng 1 câu ngắn, questions tối đa 3 câu."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"methods": methods, "chunks": batch_payload},
                    ensure_ascii=False,
                ),
            },
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    start = content.find("[")
    end = content.rfind("]") + 1
    parsed = json.loads(content[start:end])
    if not isinstance(parsed, list):
        raise ValueError("OpenAI enrichment response was not a JSON array.")
    return parsed


if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary: {summarize_chunk(sample)}\n")
    print(f"HyQA questions: {generate_hypothesis_questions(sample)}\n")
    print(f"Contextual: {contextual_prepend(sample, 'Sổ tay nhân viên')}\n")
    print(f"Auto metadata: {extract_metadata(sample)}")

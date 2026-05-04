"""Module 3: Reranking with OpenAI API."""

import json
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_MODEL, RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """OpenAI-based reranker that selects the most relevant candidate indexes."""

    def __init__(self, model_name: str = OPENAI_MODEL):
        self.model_name = model_name
        self._client = None

    def _load_model(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI reranking.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)
        return self._client

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents with one OpenAI API call."""
        if not documents:
            return []

        client = self._load_model()
        candidates = [
            {
                "index": i,
                "score": float(doc.get("score", 0.0)),
                "text": doc.get("text", "")[:900],
            }
            for i, doc in enumerate(documents)
        ]
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            max_tokens=180,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là reranker cho hệ thống RAG tiếng Việt. "
                        "Chọn các candidate trả lời câu hỏi tốt nhất. "
                        'Chỉ trả về JSON array các object: [{"index": number, "score": number}], '
                        "score từ 0 đến 1, sắp xếp giảm dần. Không thêm giải thích."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": query, "top_k": top_k, "candidates": candidates},
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        ranked_items = json.loads(content[start:end])
        results = []
        seen = set()
        for rank, item in enumerate(ranked_items[:top_k], start=1):
            index = int(item["index"])
            if index < 0 or index >= len(documents) or index in seen:
                continue
            seen.add(index)
            doc = documents[index]
            results.append(
                RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(item.get("score", 1.0 / rank)),
                    metadata=doc.get("metadata", {}),
                    rank=rank,
                )
            )
        return results


class FlashrankReranker:
    """Compatibility wrapper using the same OpenAI reranker."""

    def __init__(self):
        self._reranker = CrossEncoderReranker()

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        return self._reranker.rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": sum(times) / len(times) if times else 0,
        "min_ms": min(times) if times else 0,
        "max_ms": max(times) if times else 0,
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

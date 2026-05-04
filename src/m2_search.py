"""Module 2: Hybrid Search - BM25 + lightweight local dense search + RRF."""

import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BM25_TOP_K, COLLECTION_NAME, DENSE_TOP_K, HYBRID_TOP_K


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words, using regex tokenization if underthesea is unavailable."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")
    except Exception:
        return " ".join(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _tokens(text: str) -> list[str]:
    return segment_vietnamese(text).lower().split()


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokens(chunk.get("text", "")) for chunk in chunks]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        scores = self.bm25.get_scores(_tokens(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i].get("text", ""),
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in ranked
            if self.documents[i].get("text")
        ]


class DenseSearch:
    """In-memory TF-IDF dense-style search used as the local retrieval layer for the lab."""

    _collections: dict[str, list[dict]] = {}

    def __init__(self):
        self.documents: list[dict] = []
        self.idf: dict[str, float] = {}

    def _fit_idf(self) -> None:
        doc_count = max(len(self.documents), 1)
        df: dict[str, int] = {}
        for doc in self.documents:
            for token in set(_tokens(doc.get("text", ""))):
                df[token] = df.get(token, 0) + 1
        self.idf = {token: math.log((doc_count + 1) / (count + 1)) + 1 for token, count in df.items()}

    def _vector(self, text: str) -> dict[str, float]:
        counts: dict[str, float] = {}
        for token in _tokens(text):
            counts[token] = counts.get(token, 0.0) + 1.0
        return {token: count * self.idf.get(token, 1.0) for token, count in counts.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(value * b.get(token, 0.0) for token, value in a.items())
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks in memory."""
        self.documents = chunks
        self._fit_idf()
        for chunk in self.documents:
            chunk["_vector"] = self._vector(chunk.get("text", ""))
        self._collections[collection] = self.documents

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using cosine similarity over TF-IDF vectors."""
        docs = self._collections.get(collection, self.documents)
        if not docs:
            return []
        self.documents = docs
        if not self.idf:
            self._fit_idf()
        query_vector = self._vector(query)
        scored = []
        for doc in docs:
            score = self._cosine(query_vector, doc.get("_vector") or self._vector(doc.get("text", "")))
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(
                text=doc.get("text", ""),
                score=float(score),
                metadata={k: v for k, v in doc.get("metadata", {}).items()},
                method="dense",
            )
            for score, doc in scored[:top_k]
            if doc.get("text")
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = sum 1/(k + rank)."""
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            if result.text not in fused:
                fused[result.text] = {"score": 0.0, "result": result}
            fused[result.text]["score"] += 1.0 / (k + rank)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")

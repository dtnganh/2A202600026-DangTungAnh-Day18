"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.m1_chunking import load_documents, chunk_structure_aware
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from src.progress import ProgressBar
from config import OPENAI_API_KEY, OPENAI_MODEL, RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/4] Chunking documents...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        chunks = chunk_structure_aware(doc["text"], metadata=doc["metadata"])
        for chunk in chunks:
            all_chunks.append(
                {
                    "text": chunk.text,
                    "metadata": {
                        **chunk.metadata,
                        "parent_text": chunk.text,
                    },
                }
            )
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")

    # Step 2: Enrichment (M5)
    print("\n[2/4] Enriching chunks (M5)...")
    enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
    print(f"  Enriched {len(enriched)} chunks")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    if reranked:
        contexts = [r.metadata.get("parent_text", r.text) for r in reranked]
    else:
        contexts = [r.metadata.get("parent_text", r.text) for r in results[:3]]
    contexts = list(dict.fromkeys(contexts))

    answer = _generate_answer_openai(query, contexts)
    return answer, contexts


def _generate_answer_openai(query: str, contexts: list[str]) -> str:
    """Generate a grounded Vietnamese answer with OpenAI when configured."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for answer generation.")
    if not contexts:
        return "Không tìm thấy thông tin trong tài liệu."
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)
    context_str = "\n\n---\n\n".join(contexts[:3])[:7000]
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        max_tokens=220,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý RAG. Trả lời bằng tiếng Việt, ngắn gọn, chỉ dựa trên context. "
                    "Nếu context không có thông tin, trả lời: Không tìm thấy thông tin trong tài liệu. "
                    "Ưu tiên câu trả lời trực tiếp, số liệu và danh sách khi câu hỏi yêu cầu."
                ),
            },
            {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _extract_answer(query: str, contexts: list[str]) -> str:
    """Pick the most relevant sentence from retrieved contexts."""
    if not contexts:
        return "Không tìm thấy thông tin."

    import re

    query_terms = set(re.findall(r"[\wÀ-ỹ]+", query.lower(), flags=re.UNICODE))
    sentences = []
    for context in contexts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", context):
            sentence = sentence.strip()
            if sentence and not sentence.startswith("#"):
                sentences.append(sentence)

    def score(sentence: str) -> tuple[int, int]:
        terms = set(re.findall(r"[\wÀ-ỹ]+", sentence.lower(), flags=re.UNICODE))
        numeric_bonus = 2 if re.search(r"\d+", sentence) else 0
        lower_query = query.lower()
        lower_sentence = sentence.lower()
        intent_bonus = 0
        if "là gì" in lower_query and " là " in f" {lower_sentence} ":
            intent_bonus += 3
        if ("gồm" in lower_query or "ví dụ" in lower_query) and ("gồm" in lower_sentence or "ví dụ" in lower_sentence):
            intent_bonus += 3
        if "quyền" in lower_query and ("các quyền" in lower_sentence or "quyền được biết" in lower_sentence):
            intent_bonus += 4
        return (len(query_terms & terms) + numeric_bonus + intent_bonus, -len(sentence))

    best = max(sentences, key=score, default=contexts[0])
    return best


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    if OPENAI_API_KEY:
        print(f"  OpenAI generation enabled: {OPENAI_MODEL}")
    else:
        raise RuntimeError("OPENAI_API_KEY is required; offline fallback is disabled.")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []
    progress = ProgressBar("  Eval queries", len(test_set))

    for i, item in enumerate(test_set):
        progress.update(i, f"query {i + 1}/{len(test_set)}")
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        progress.update(i + 1, f"query {i + 1}/{len(test_set)} done")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        marker = "PASS" if s >= 0.75 else "LOW"
        print(f"  [{marker}] {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")

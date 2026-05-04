"""Module 4: RAGAS-style evaluation with deterministic local metrics."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    stopwords = {
        "là",
        "và",
        "của",
        "có",
        "cho",
        "trong",
        "được",
        "về",
        "các",
        "một",
        "những",
        "the",
        "a",
    }
    return {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if token not in stopwords and len(token) > 1
    }


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Return RAGAS-compatible metrics, using deterministic local scoring."""
    per_question: list[EvalResult] = []

    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths, strict=False):
        context_text = " ".join(ctxs)
        faithfulness = _overlap(answer, context_text) if answer else 0.0
        answer_relevancy = (_overlap(question, answer) + _overlap(ground_truth, answer)) / 2
        context_precision = _overlap(question, context_text)
        context_recall = _overlap(ground_truth, context_text)
        per_question.append(
            EvalResult(
                question=question,
                answer=answer,
                contexts=ctxs,
                ground_truth=ground_truth,
                faithfulness=_clamp(faithfulness),
                answer_relevancy=_clamp(answer_relevancy),
                context_precision=_clamp(context_precision),
                context_recall=_clamp(context_recall),
            )
        )

    def avg(metric: str) -> float:
        if not per_question:
            return 0.0
        return sum(getattr(result, metric) for result in per_question) / len(per_question)

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using a diagnostic mapping."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def avg_score(result: EvalResult) -> float:
        return sum(getattr(result, metric) for metric in metric_names) / len(metric_names)

    failures = []
    for result in sorted(eval_results, key=avg_score)[:bottom_n]:
        scores = {metric: getattr(result, metric) for metric in metric_names}
        worst_metric = min(scores, key=scores.get)
        score = scores[worst_metric]
        if worst_metric == "faithfulness":
            diagnosis = "LLM hallucinating or answer not grounded in context"
            suggested_fix = "Tighten prompt, quote context, lower temperature"
        elif worst_metric == "context_recall":
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking, add BM25 terms, or increase top_k"
        elif worst_metric == "context_precision":
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking, metadata filters, or better query matching"
        else:
            diagnosis = "Answer does not match the question"
            suggested_fix = "Improve answer generation prompt or extractive answer selection"

        failures.append(
            {
                "question": result.question,
                "expected": result.ground_truth,
                "got": result.answer,
                "worst_metric": worst_metric,
                "score": float(score),
                "avg_score": float(avg_score(result)),
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "per_question": [asdict(item) for item in results.get("per_question", [])],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.progress import ProgressBar


def main():
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()
    stage_progress = ProgressBar("Main stages", 3)

    os.makedirs("reports", exist_ok=True)

    # Step 1: Basic Baseline
    stage_progress.update(0, "baseline")
    print("\nSTEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline
    run_baseline()
    stage_progress.update(1, "baseline done")

    # Step 2: Production Pipeline
    print("\nSTEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline
    search, reranker = build_pipeline()
    prod_results = evaluate_pipeline(search, reranker)
    stage_progress.update(2, "production done")

    # Move reports to reports/
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f):
            os.replace(f, f"reports/{f}")

    # Step 3: Comparison
    print("\nSTEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Delta':>8}")
        print("-" * 55)
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "OK" if p >= 0.75 else "--"
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")
    stage_progress.update(3, "complete")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Reports are ready in reports/. Run check_lab.py for the final submission check.")


if __name__ == "__main__":
    main()

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import fixed_chunks, header_aware_chunks, load_corpus
from src.grounding import NLIVerifier, should_refuse
from src.retrievers import BM25Retriever, CrossEncoderReranker, DenseFAISSRetriever, HybridRRFRetriever

PREMISE_THRESHOLD = 0.55  # matches RagPipeline.premise_threshold

ROOT = Path(__file__).resolve().parents[1]

# one threshold per method because the score scales are not comparable.
# these were picked on this query set and are reported as tuned, not universal.
THRESHOLDS = {
    "bm25": 2.0,
    "dense_faiss": 0.25,
    "hybrid_rrf": 0.020,
    "hybrid_rrf_rerank_top20": 0.35,
    "hybrid_rrf_rerank_top8": 0.35,
}
THRESHOLDS_LIGHTWEIGHT = {
    "bm25": 2.0,
    "dense_faiss": 0.08,
    "hybrid_rrf": 0.020,
    "hybrid_rrf_rerank_top20": 0.15,
    "hybrid_rrf_rerank_top8": 0.15,
}


def load_queries() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval/queries.jsonl").read_text().splitlines() if line.strip()]


def evaluate(search_fn, queries: list[dict], threshold: float) -> dict:
    hits, rr, latencies = [], [], []
    refusal_correct, oos_total, oos_refused = 0, 0, 0
    by_type: dict[str, list[int]] = {}
    answerable_refused = 0
    per_query = []
    for q in queries:
        start = time.perf_counter()
        results = search_fn(q["query"], 5)
        latencies.append((time.perf_counter() - start) * 1000)
        top_score = results[0]["score"] if results else 0.0
        refused = should_refuse(top_score, threshold)
        if q["answerable"]:
            expected_file = q["expected_source"].split("#")[0]
            ranks = [i for i, r in enumerate(results, start=1) if r["source_file"] == expected_file]
            hits.append(1.0 if ranks and ranks[0] <= 3 else 0.0)
            rr.append(1.0 / ranks[0] if ranks else 0.0)
            refusal_correct += 0 if refused else 1
            answerable_refused += 1 if refused else 0
            per_query.append({"id": q["id"], "hit3": hits[-1], "rr": round(rr[-1], 3), "refused": refused})
        else:
            oos_total += 1
            oos_refused += 1 if refused else 0
            refusal_correct += 1 if refused else 0
            by_type.setdefault(q.get("unanswerable_type", "out_of_scope"), []).append(1 if refused else 0)
            per_query.append({"id": q["id"], "hit3": None, "rr": None, "refused": refused})
    n_answerable = sum(1 for q in queries if q["answerable"])
    return {
        "hit_at_3": round(float(np.mean(hits)), 3) if hits else 0.0,
        "mrr_at_5": round(float(np.mean(rr)), 3) if rr else 0.0,
        "refusal_accuracy": round(refusal_correct / len(queries), 3),
        "oos_refusal_rate": round(oos_refused / oos_total, 3) if oos_total else 0.0,
        "oos_refusal_accuracy": round(float(np.mean(by_type["out_of_scope"])), 3) if by_type.get("out_of_scope") else 0.0,
        "false_premise_refusal_accuracy": round(float(np.mean(by_type["false_premise"])), 3) if by_type.get("false_premise") else 0.0,
        "answerable_false_refusal_rate": round(answerable_refused / n_answerable, 3) if n_answerable else 0.0,
        "median_latency_ms": round(float(np.median(latencies)), 1),
        "n_answerable": n_answerable,
        "per_query": per_query,
    }


def evaluate_gate(search_fn, queries: list[dict], threshold: float, verifier: NLIVerifier) -> dict:
    # the full pipeline gate on top of one retrieval method: check 1 is the
    # score gate, check 2 is the NLI premise check against the top chunk.
    # also times the NLI calls so the added CPU latency is measured, not guessed.
    by_type: dict[str, list[int]] = {}
    answerable_refused, correct, nli_latencies = 0, 0, []
    per_query = []
    for q in queries:
        results = search_fn(q["query"], 5)
        top_score = results[0]["score"] if results else 0.0
        refused = should_refuse(top_score, threshold)
        reason = "out_of_scope" if refused else None
        if not refused and results:
            start = time.perf_counter()
            probs = verifier.probs(results[0]["text"], q["query"])
            nli_latencies.append((time.perf_counter() - start) * 1000)
            if probs.get("contradiction", 0.0) >= PREMISE_THRESHOLD:
                refused, reason = True, "false_premise"
        if q["answerable"]:
            answerable_refused += 1 if refused else 0
            correct += 0 if refused else 1
        else:
            by_type.setdefault(q.get("unanswerable_type", "out_of_scope"), []).append(1 if refused else 0)
            correct += 1 if refused else 0
        per_query.append({"id": q["id"], "refused": refused, "refusal_reason": reason})
    n_answerable = sum(1 for q in queries if q["answerable"])
    return {
        "refusal_accuracy": round(correct / len(queries), 3),
        "oos_refusal_accuracy": round(float(np.mean(by_type["out_of_scope"])), 3) if by_type.get("out_of_scope") else 0.0,
        "false_premise_refusal_accuracy": round(float(np.mean(by_type["false_premise"])), 3) if by_type.get("false_premise") else 0.0,
        "answerable_false_refusal_rate": round(answerable_refused / n_answerable, 3) if n_answerable else 0.0,
        "nli_median_latency_ms": round(float(np.median(nli_latencies)), 1) if nli_latencies else 0.0,
        "nli_checks": len(nli_latencies),
        "per_query": per_query,
    }


def build_methods(chunks, lightweight: bool) -> dict:
    bm25 = BM25Retriever(chunks)
    dense = DenseFAISSRetriever(chunks, lightweight=lightweight)
    hybrid = HybridRRFRetriever(bm25, dense)
    reranker = CrossEncoderReranker(lightweight=lightweight)
    return {
        "bm25": lambda q, k: bm25.search(q, top_k=k),
        "dense_faiss": lambda q, k: dense.search(q, top_k=k),
        "hybrid_rrf": lambda q, k: hybrid.search(q, top_k=k),
        "hybrid_rrf_rerank_top20": lambda q, k: reranker.rerank(q, hybrid.search(q, top_k=20, pool=20), top_k=k),
        "hybrid_rrf_rerank_top8": lambda q, k: reranker.rerank(q, hybrid.search(q, top_k=20, pool=20)[:8], top_k=k),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightweight", action="store_true", help="TF-IDF dense mode; numbers are not comparable to the real run")
    args = parser.parse_args()
    thresholds = THRESHOLDS_LIGHTWEIGHT if args.lightweight else THRESHOLDS

    queries = load_queries()
    docs = load_corpus(ROOT / "corpus")
    header_chunks = header_aware_chunks(docs)
    fixed = fixed_chunks(docs)
    print(f"header-aware chunks: {len(header_chunks)}, fixed chunks: {len(fixed)}")

    methods = build_methods(header_chunks, args.lightweight)
    results = {}
    for name, fn in methods.items():
        results[name] = evaluate(fn, queries, thresholds[name])
        r = results[name]
        print(f"{name:20s} hit@3={r['hit_at_3']:.3f} mrr@5={r['mrr_at_5']:.3f} refusal_acc={r['refusal_accuracy']:.3f} oos_refusal={r['oos_refusal_rate']:.3f} latency_ms={r['median_latency_ms']}")

    # chunking comparison on the hybrid, everything else held fixed
    chunking_results = {}
    for label, chunks in [("header_aware", header_chunks), ("fixed", fixed)]:
        hybrid = build_methods(chunks, args.lightweight)["hybrid_rrf"]
        chunking_results[label] = evaluate(hybrid, queries, thresholds["hybrid_rrf"])
        r = chunking_results[label]
        print(f"chunking={label:14s} hit@3={r['hit_at_3']:.3f} mrr@5={r['mrr_at_5']:.3f} chunks={len(chunks)}")

    # full gate calibration on the default pipeline method (rerank, top-8 pool),
    # plus the per-claim NLI latency measured on the saved sample audit
    verifier = NLIVerifier(lightweight=args.lightweight)
    gate = evaluate_gate(methods["hybrid_rrf_rerank_top8"], queries, thresholds["hybrid_rrf_rerank_top8"], verifier)
    print(f"gate(top8+premise)   refusal_acc={gate['refusal_accuracy']:.3f} oos_acc={gate['oos_refusal_accuracy']:.3f} fp_acc={gate['false_premise_refusal_accuracy']:.3f} ans_false_refusal={gate['answerable_false_refusal_rate']:.3f} nli_ms={gate['nli_median_latency_ms']}")
    sample = json.loads((ROOT / "eval/sample_audit.json").read_text())
    chunk_by_citation = {(c.source_file, c.section): c.text for c in header_chunks}
    claim_latencies = []
    for claim in sample["claims"]:
        text = chunk_by_citation.get((claim["citation"]["source_file"], claim["citation"]["section"]))
        if text:
            start = time.perf_counter()
            verifier.probs(text, claim["claim"])
            claim_latencies.append((time.perf_counter() - start) * 1000)
    claim_nli_ms = round(float(np.median(claim_latencies)), 1) if claim_latencies else 0.0
    print(f"nli claim-pair median latency: {claim_nli_ms} ms over {len(claim_latencies)} claims")

    mode = "lightweight TF-IDF dense mode (not the numbers reported in the readme)" if args.lightweight else "sentence-transformers/all-MiniLM-L6-v2 embeddings, cross-encoder/ms-marco-MiniLM-L-6-v2 reranker, CPU"
    n_ans = sum(1 for q in queries if q["answerable"])
    n_oos = sum(1 for q in queries if q.get("unanswerable_type") == "out_of_scope")
    n_fp = sum(1 for q in queries if q.get("unanswerable_type") == "false_premise")
    lines = [
        "# Ablation results",
        "",
        f"Mode: {mode}.",
        f"Queries: {len(queries)} ({n_ans} answerable, {n_oos} out of scope, {n_fp} false premise). Header-aware chunks: {len(header_chunks)}. Fixed chunks: {len(fixed)}.",
        "Latency is median per-query retrieval time on the machine that ran this script, single thread, warm index.",
        "Refusal here is the score gate only (check 1); the full pipeline also runs the false-premise NLI check, measured separately in the gate calibration section.",
        "",
        "## Retrievers (header-aware chunking)",
        "",
        "| method | Hit@3 | MRR@5 | refusal accuracy | out-of-scope refusal accuracy | false-premise refusal accuracy | answerable false-refusal rate | median latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['hit_at_3']:.3f} | {r['mrr_at_5']:.3f} | {r['refusal_accuracy']:.3f} | {r['oos_refusal_accuracy']:.3f} | {r['false_premise_refusal_accuracy']:.3f} | {r['answerable_false_refusal_rate']:.3f} | {r['median_latency_ms']} |")
    lines += [
        "",
        "## Chunking (hybrid_rrf, everything else fixed)",
        "",
        "| chunking | chunks | Hit@3 | MRR@5 |",
        "|---|---|---|---|",
    ]
    for label, chunks in [("header_aware", header_chunks), ("fixed", fixed)]:
        r = chunking_results[label]
        lines.append(f"| {label} | {len(chunks)} | {r['hit_at_3']:.3f} | {r['mrr_at_5']:.3f} |")
    lines += [
        "",
        "## Gate calibration (hybrid_rrf_rerank_top8 + premise NLI check)",
        "",
        f"Check 1 threshold {thresholds['hybrid_rrf_rerank_top8']}, premise contradiction threshold {PREMISE_THRESHOLD}.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| refusal accuracy (all queries) | {gate['refusal_accuracy']:.3f} |",
        f"| out-of-scope refusal accuracy (n={n_oos}) | {gate['oos_refusal_accuracy']:.3f} |",
        f"| false-premise refusal accuracy (n={n_fp}) | {gate['false_premise_refusal_accuracy']:.3f} |",
        f"| answerable false-refusal rate (n={n_ans}) | {gate['answerable_false_refusal_rate']:.3f} |",
        f"| NLI premise check median latency (ms, {gate['nli_checks']} checks) | {gate['nli_median_latency_ms']} |",
        f"| NLI claim-pair median latency (ms, sample audit) | {claim_nli_ms} |",
        "",
        "## Per-query detail (hybrid_rrf_rerank_top8, score gate only)",
        "",
        "| query | Hit@3 | reciprocal rank | refused |",
        "|---|---|---|---|",
    ]
    for row in results["hybrid_rrf_rerank_top8"]["per_query"]:
        lines.append(f"| {row['id']} | {row['hit3']} | {row['rr']} | {row['refused']} |")
    lines += ["", "## Per-query gate detail (score gate + premise check)", "", "| query | refused | refusal_reason |", "|---|---|---|"]
    for row in gate["per_query"]:
        lines.append(f"| {row['id']} | {row['refused']} | {row['refusal_reason']} |")
    lines.append("")
    (ROOT / "eval/ablation_results.md").write_text("\n".join(lines))
    (ROOT / "eval/ablation_results.json").write_text(json.dumps({"results": results, "chunking": chunking_results, "gate": gate, "nli_claim_median_latency_ms": claim_nli_ms}, indent=2))
    print("wrote eval/ablation_results.md")


if __name__ == "__main__":
    main()

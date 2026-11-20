# Ablation results

Mode: sentence-transformers/all-MiniLM-L6-v2 embeddings, cross-encoder/ms-marco-MiniLM-L-6-v2 reranker, CPU.
Queries: 20 (14 answerable, 6 out of scope). Header-aware chunks: 29. Fixed chunks: 28.
Latency is median per-query retrieval time on the machine that ran this script, single thread, warm index.

## Retrievers (header-aware chunking)

| method | Hit@3 | MRR@5 | refusal accuracy | out-of-scope refusal rate | median latency (ms) |
|---|---|---|---|---|---|
| bm25 | 1.000 | 0.952 | 0.700 | 0.000 | 0.3 |
| dense_faiss | 1.000 | 1.000 | 0.800 | 0.333 | 23.6 |
| hybrid_rrf | 1.000 | 1.000 | 0.700 | 0.000 | 12.4 |
| hybrid_rrf_rerank | 1.000 | 0.964 | 0.800 | 0.667 | 2594.6 |

## Chunking (hybrid_rrf, everything else fixed)

| chunking | chunks | Hit@3 | MRR@5 |
|---|---|---|---|
| header_aware | 29 | 1.000 | 1.000 |
| fixed | 28 | 1.000 | 1.000 |

## Per-query detail (hybrid_rrf_rerank)

| query | Hit@3 | reciprocal rank | refused |
|---|---|---|---|
| Q01 | 1.0 | 1.0 | False |
| Q02 | 1.0 | 1.0 | False |
| Q03 | 1.0 | 0.5 | True |
| Q04 | 1.0 | 1.0 | False |
| Q05 | 1.0 | 1.0 | False |
| Q06 | 1.0 | 1.0 | False |
| Q07 | 1.0 | 1.0 | False |
| Q08 | 1.0 | 1.0 | False |
| Q09 | 1.0 | 1.0 | True |
| Q10 | 1.0 | 1.0 | False |
| Q11 | 1.0 | 1.0 | False |
| Q12 | 1.0 | 1.0 | False |
| Q13 | 1.0 | 1.0 | False |
| Q14 | 1.0 | 1.0 | False |
| Q15 | None | None | True |
| Q16 | None | None | True |
| Q17 | None | None | False |
| Q18 | None | None | True |
| Q19 | None | None | True |
| Q20 | None | None | False |

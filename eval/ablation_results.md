# Ablation results

Mode: sentence-transformers/all-MiniLM-L6-v2 embeddings, cross-encoder/ms-marco-MiniLM-L-6-v2 reranker, CPU.
Queries: 60 (36 answerable, 14 out of scope, 10 false premise). Header-aware chunks: 303. Fixed chunks: 294.
Latency is median per-query retrieval time on the machine that ran this script, single thread, warm index.
Refusal here is the score gate only (check 1); the full pipeline also runs the false-premise NLI check, measured separately in the gate calibration section.

## Retrievers (header-aware chunking)

| method | Hit@3 | MRR@5 | refusal accuracy | out-of-scope refusal accuracy | false-premise refusal accuracy | answerable false-refusal rate | median latency (ms) |
|---|---|---|---|---|---|---|---|
| bm25 | 0.944 | 0.931 | 0.600 | 0.000 | 0.000 | 0.000 | 2.6 |
| dense_faiss | 0.917 | 0.881 | 0.700 | 0.429 | 0.000 | 0.000 | 102.4 |
| hybrid_rrf | 0.917 | 0.914 | 0.617 | 0.071 | 0.000 | 0.000 | 110.3 |
| hybrid_rrf_rerank_top20 | 0.944 | 0.881 | 0.800 | 1.000 | 0.000 | 0.056 | 4310.5 |
| hybrid_rrf_rerank_top8 | 0.944 | 0.885 | 0.800 | 1.000 | 0.000 | 0.056 | 1892.0 |

## Chunking (hybrid_rrf, everything else fixed)

| chunking | chunks | Hit@3 | MRR@5 |
|---|---|---|---|
| header_aware | 303 | 0.917 | 0.914 |
| fixed | 294 | 0.972 | 0.907 |

## Gate calibration (hybrid_rrf_rerank_top8 + premise NLI check)

Check 1 threshold 0.35, premise contradiction threshold 0.55.

| metric | value |
|---|---|
| refusal accuracy (all queries) | 0.833 |
| out-of-scope refusal accuracy (n=14) | 1.000 |
| false-premise refusal accuracy (n=10) | 0.300 |
| answerable false-refusal rate (n=36) | 0.083 |
| NLI premise check median latency (ms, 44 checks) | 1445.2 |
| NLI claim-pair median latency (ms, sample audit) | 813.5 |

## Per-query detail (hybrid_rrf_rerank_top8, score gate only)

| query | Hit@3 | reciprocal rank | refused |
|---|---|---|---|
| Q01 | 1.0 | 1.0 | False |
| Q02 | 1.0 | 1.0 | False |
| Q03 | 0.0 | 0.0 | True |
| Q04 | 1.0 | 1.0 | False |
| Q05 | 1.0 | 1.0 | False |
| Q06 | 1.0 | 0.333 | False |
| Q07 | 1.0 | 0.5 | False |
| Q08 | 1.0 | 1.0 | False |
| Q09 | 1.0 | 0.5 | True |
| Q10 | 1.0 | 1.0 | False |
| Q11 | 0.0 | 0.2 | False |
| Q12 | 1.0 | 1.0 | False |
| Q13 | 1.0 | 1.0 | False |
| Q14 | 1.0 | 1.0 | False |
| Q15 | None | None | True |
| Q16 | None | None | True |
| Q17 | None | None | False |
| Q18 | None | None | True |
| Q19 | None | None | True |
| Q20 | None | None | False |
| Q21 | 1.0 | 1.0 | False |
| Q22 | 1.0 | 1.0 | False |
| Q23 | 1.0 | 1.0 | False |
| Q24 | 1.0 | 1.0 | False |
| Q25 | 1.0 | 1.0 | False |
| Q26 | 1.0 | 1.0 | False |
| Q27 | 1.0 | 1.0 | False |
| Q28 | 1.0 | 1.0 | False |
| Q29 | 1.0 | 1.0 | False |
| Q30 | 1.0 | 1.0 | False |
| Q31 | 1.0 | 1.0 | False |
| Q32 | 1.0 | 1.0 | False |
| Q33 | 1.0 | 1.0 | False |
| Q34 | 1.0 | 1.0 | False |
| Q35 | 1.0 | 0.333 | False |
| Q36 | 1.0 | 1.0 | False |
| Q37 | 1.0 | 1.0 | False |
| Q38 | 1.0 | 1.0 | False |
| Q39 | 1.0 | 1.0 | False |
| Q40 | 1.0 | 1.0 | False |
| Q41 | 1.0 | 1.0 | False |
| Q42 | 1.0 | 1.0 | False |
| Q43 | None | None | True |
| Q44 | None | None | True |
| Q45 | None | None | True |
| Q46 | None | None | True |
| Q47 | None | None | True |
| Q48 | None | None | True |
| Q49 | None | None | True |
| Q50 | None | None | True |
| Q51 | None | None | True |
| Q52 | None | None | True |
| Q53 | None | None | False |
| Q54 | None | None | False |
| Q55 | None | None | False |
| Q56 | None | None | False |
| Q57 | None | None | False |
| Q58 | None | None | False |
| Q59 | None | None | False |
| Q60 | None | None | False |

## Per-query gate detail (score gate + premise check)

| query | refused | refusal_reason |
|---|---|---|
| Q01 | False | None |
| Q02 | False | None |
| Q03 | True | out_of_scope |
| Q04 | False | None |
| Q05 | False | None |
| Q06 | False | None |
| Q07 | True | false_premise |
| Q08 | False | None |
| Q09 | True | out_of_scope |
| Q10 | False | None |
| Q11 | False | None |
| Q12 | False | None |
| Q13 | False | None |
| Q14 | False | None |
| Q15 | True | out_of_scope |
| Q16 | True | out_of_scope |
| Q17 | False | None |
| Q18 | True | out_of_scope |
| Q19 | True | out_of_scope |
| Q20 | False | None |
| Q21 | False | None |
| Q22 | False | None |
| Q23 | False | None |
| Q24 | False | None |
| Q25 | False | None |
| Q26 | False | None |
| Q27 | False | None |
| Q28 | False | None |
| Q29 | False | None |
| Q30 | False | None |
| Q31 | False | None |
| Q32 | False | None |
| Q33 | False | None |
| Q34 | False | None |
| Q35 | False | None |
| Q36 | False | None |
| Q37 | False | None |
| Q38 | False | None |
| Q39 | False | None |
| Q40 | False | None |
| Q41 | False | None |
| Q42 | False | None |
| Q43 | True | out_of_scope |
| Q44 | True | out_of_scope |
| Q45 | True | out_of_scope |
| Q46 | True | out_of_scope |
| Q47 | True | out_of_scope |
| Q48 | True | out_of_scope |
| Q49 | True | out_of_scope |
| Q50 | True | out_of_scope |
| Q51 | True | out_of_scope |
| Q52 | True | out_of_scope |
| Q53 | False | None |
| Q54 | True | false_premise |
| Q55 | False | None |
| Q56 | True | false_premise |
| Q57 | False | None |
| Q58 | False | None |
| Q59 | False | None |
| Q60 | True | false_premise |

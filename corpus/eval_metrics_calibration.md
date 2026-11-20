# Evaluation metrics and calibration

## Hit@K and MRR

Hit@K is the fraction of queries where at least one relevant document
appears in the top K results. It ignores position inside K, so Hit@3 of 0.9
can hide a system that always ranks the right document third.

MRR (mean reciprocal rank) averages 1 / rank of the first relevant result
over queries, usually reported at a cutoff such as MRR@5 or MRR@10. A query
whose first relevant result is at rank 2 contributes 0.5. MRR is the better
single number when users mostly read the first result or two, which is the
case for retrieval feeding an LLM context window.

## NDCG

NDCG@K handles graded relevance. DCG sums (2^rel - 1) / log2(rank + 1) over
the top K, and NDCG divides by the DCG of the ideal ordering. Use it when
relevance is not binary (a section that fully answers the query vs. one that
mentions the term in passing). With binary labels and one relevant document
per query, NDCG adds little over MRR, and reporting both is mostly noise.

## Cohen's kappa

Cohen's kappa measures agreement between two raters corrected for the
agreement expected by chance: kappa = (p_o - p_e) / (1 - p_e). A kappa of
0.61 to 0.80 is usually called substantial agreement. For ordinal scores
(1 to 5 rubric grades), quadratic weighted kappa penalizes a 2-point
disagreement more than a 1-point one and is the version to report. Raw
percent agreement on its own is misleading when the label distribution is
skewed, which it almost always is in hallucination audits.

## Faithfulness vs groundedness

Faithfulness asks whether an answer is consistent with the retrieved
context. Groundedness is stricter and more operational: can each factual
claim in the answer be pointed at a specific source passage? An answer can
be faithful in tone and still contain a claim with no source, for example a
plausible number that appears in no chunk. Claim-level checks are more
expensive than one score for the whole answer, but they tell you which
sentence to delete, which a single faithfulness score never does.

## Thresholds and abstention

A retrieval system that must sometimes say "I don't know" needs a score
threshold, and the threshold is a trade-off, not a constant of nature. Set
it too high and in-scope questions get refused; too low and out-of-scope
questions get confident answers stitched together from irrelevant chunks.
Report the refusal rate on out-of-scope queries and the false-refusal rate
on in-scope queries separately. A single accuracy number over both hides
exactly the failure you care about.

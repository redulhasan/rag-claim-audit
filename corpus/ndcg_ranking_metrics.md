# Ranking metrics: NDCG, MAP, and ERR

## DCG and NDCG

For a ranked list with graded relevance rel_i at position i, discounted
cumulative gain is DCG = sum_i (2^{rel_i} - 1) / log2(i + 1). The numerator
(gain) grows exponentially in the relevance grade, so a highly relevant
result is worth much more than two marginally relevant ones; the denominator
discounts by position, with rank 1 undiscounted (log2(2) = 1), rank 2
divided by about 1.58, rank 9 by about 3.32. NDCG divides DCG by the ideal
DCG (the DCG of the results sorted by relevance), giving a score in [0, 1]
comparable across queries with different numbers of relevant documents.
Worked example: relevance grades [3, 1, 2] at ranks 1 to 3 give DCG =
7/1 + 1/1.58 + 3/2 = about 9.13; the ideal order [3, 2, 1] gives
7 + 3/1.58 + 1/2 = about 9.40, so NDCG is about 0.972.

## Graded relevance is the input cost

NDCG needs graded judgments (typically 0 to 3), which cost more to collect
than binary labels and carry annotator disagreement on the middle grades.
With only binary labels NDCG still works (grades 0 and 1) but behaves much
like a smoothed precision at rank. The metric choice should follow the
labels you actually have: binary labels plus a need to reward the first hit
points to MRR; binary labels plus a need to count all relevant results
points to MAP or recall at k; graded labels point to NDCG. Reporting NDCG
computed from binary labels as if it carried graded information is a common
small dishonesty in eval write-ups.

## MAP versus MRR

Mean average precision averages, over queries, the precision at each rank
where a relevant document appears, divided by the total number of relevant
documents for that query. It therefore rewards retrieving all relevant
documents early and penalizes missing ones entirely (a relevant document
never retrieved contributes 0). MRR only cares about the first relevant
result. The two diverge exactly when queries have multiple relevant
documents whose recall matters: known-item search (one right answer) is an
MRR task; literature review (find everything on a topic) is a MAP or recall
task. Choosing MRR for a multi-answer task makes systems that stop after one
good hit look perfect.

## ERR and cascade models

Expected reciprocal rank models the user scanning down the list and stopping
when satisfied: ERR = sum_i (1/i) * P(user stops at i), where the stop
probability at i depends on the relevance grade through
R_i = (2^{rel_i} - 1) / 2^{rel_max}, and the probability of reaching i is the
product of not stopping earlier. ERR correlates better with session-level
satisfaction in web search studies than NDCG in some comparisons, but it is
rarely reported outside IR research because it needs graded labels and its
absolute values are unintuitive. It is included here because any serious
metric survey that stops at NDCG is incomplete.

## Ties, cutoffs, and aggregation traps

Three traps recur. Ties: systems returning identical scores for many
documents make the metric depend on tie-breaking order; report the expected
metric under random tie-breaking or break ties pessimistically. Cutoffs:
NDCG@5 and NDCG@10 can rank systems differently, so the cutoff must be
chosen from the product surface (how many results a user actually sees) and
fixed before the experiment. Aggregation: averaging NDCG over queries hides
variance; per-query deltas between two systems, with a paired significance
test (randomization test is the IR standard), are the honest comparison. A
0.01 average NDCG gain with per-query wins and losses split evenly is not a
gain.

## Offline metrics versus online outcomes

Offline ranking metrics predict, imperfectly, online behavior: interleaving
experiments and A/B tests regularly contradict offline NDCG orderings when
the offline judgments are stale or the label definition differs from user
satisfaction. The metrics are still the right tool for regression testing
and model iteration, where the comparison is same-queries, same-labels,
different-system. They are the wrong tool for forecasting click or
satisfaction deltas. State which job a reported metric is doing.

## Choosing the cutoff from the product surface

The cutoff k in NDCG@k should be the number of results a user or a
downstream component actually consumes. For a RAG system that feeds the top
5 chunks to a generator, NDCG@5 and Hit@5 describe the system; NDCG@100
describes a system nobody runs. Reporting several cutoffs is fine; choosing
the cutoff after seeing which one flatters the system is not. The same
logic fixes MRR's truncation (MRR@10 is standard because ranks past 10 are
rarely seen) and recall's denominator. Write the cutoff choice in the eval
plan before running, with the product reason, and the metric section of the
report writes itself.

## Significance testing practice

Per-query metric differences between two systems are paired data; the
standard tests are the randomization (permutation) test on per-query
deltas, or a paired bootstrap over queries. With 50 queries, an average
NDCG difference under about 0.02 is usually indistinguishable from noise;
with 500 queries, differences of 0.005 can be real. This arithmetic should
govern eval set size: decide the smallest difference worth acting on, then
collect enough judged queries to detect it. Most internal eval sets are
underpowered for the decisions made from them, which is a fixable problem:
it costs judgments, not insight.

## Collecting the labels

Graded relevance labels need guidelines with examples per grade, double
labeling on a sample to measure annotator agreement (report it; kappa below
about 0.6 on the relevance task means the guidelines, not the annotators,
need work), and adjudication for the disagreements that remain. Pooling is
the standard way to build the judged set: run all systems being compared,
judge the union of their top-k results, and treat unjudged documents as
non-relevant, which biases against systems that retrieve novel documents,
a known and accepted limitation as long as all compared systems contributed
to the pool. A new system evaluated later against an old pool is
systematically underscored, and forgetting this has killed more than one
honest improvement.

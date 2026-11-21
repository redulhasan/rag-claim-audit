# Speculative decoding variants: Medusa, EAGLE, and lookahead

## The base algorithm and its arithmetic

A small draft model proposes gamma tokens; the target model scores all of
them in one parallel forward pass; tokens are accepted by rejection sampling
that keeps the target's distribution exact (accept a draft token x with
probability min(1, p(x) / q(x)), resample from the adjusted distribution on
the first rejection, and always keep one extra target-sampled token when all
gamma are accepted). If each draft token is accepted independently with
probability alpha, the expected tokens produced per target forward pass are
(1 - alpha^(gamma+1)) / (1 - alpha). At alpha = 0.7 and gamma = 4 that is
(1 - 0.168) / 0.3, about 2.77 tokens per pass instead of 1. The speedup is
that factor divided by the cost of the pass, which grows slightly with gamma
plus the draft model's gamma sequential steps.

## Why acceptance rate is the whole game

Wall-clock speedup is roughly (expected tokens per pass) / (1 + gamma *
draft_cost_ratio + verification overhead). A draft model that is 10x cheaper
but has alpha = 0.5 loses to one that is 5x cheaper with alpha = 0.75 on most
hardware. Acceptance rate is highest when draft and target were trained on
similar data and the task is predictable (code, templated text) and lowest on
open-ended generation at high temperature. Reported end-to-end speedups for
well-matched draft models are 2x to 3x at batch size 1; at large batch sizes
the target forward pass is compute bound, verification of gamma extra tokens
costs real FLOPs, and speculative decoding can be a net loss, which is why
serving engines make it adaptive or disable it under load.

## Medusa: draft heads instead of a draft model

Medusa removes the separate draft model: k extra prediction heads on top of
the target model's final hidden state predict tokens at positions +2, +3,
and beyond, trained on the model's own generations or instruction data while
the base model stays frozen. The heads are cheap MLPs, so drafting costs one
shared forward pass. Because the heads share the target's representations,
acceptance is decent but below a dedicated draft model; the candidates from
multiple heads form a tree (typical acceptance lengths around 2 to 3 tokens
per step in the paper) verified with tree attention in one pass. Medusa's
appeal is operational: no second model to serve, align, or keep in memory.

## EAGLE: drafting in feature space

EAGLE drafts by predicting the target model's next hidden feature (the input
to the LM head) autoregressively, then sampling the token from that feature.
Features are smoother and more predictable than token ids, so a tiny
one-layer drafter conditioned on the feature sequence and the sampled tokens
reaches higher acceptance than token-level drafters of the same size; EAGLE
reports about 2.7x to 3.2x speedup on chat workloads at batch 1, roughly 0.5
to 1 token more accepted per step than Medusa-class heads. EAGLE-2 makes the
candidate tree dynamic, expanding branches by the drafter's confidence.
EAGLE-3 drops the feature-prediction constraint and trains the drafter on
multiple layers' features, gaining further acceptance. All variants keep the
same exactness guarantee through the verification step.

## Tree attention verification

Verifying a tree of candidates in one pass needs an attention mask where
each node attends only to its ancestors. Implementations build this mask per
step (or use flattened tree encodings) and gather the target logits at every
node; the accepted path is the longest prefix path the rejection rule
approves. The tree budget (total nodes per step, often 32 to 64) trades
verification FLOPs against expected accepted length. This is why speculative
decoding implementations report both the tree size and the acceptance length:
speedup claims without them are not reproducible.

## Self-speculative and lookahead methods

Self-speculative decoding uses the target model itself as the drafter by
skipping layers: draft with the first fraction of layers plus the LM head,
verify with the full model. No extra parameters and no distribution mismatch
in vocabulary, but the draft quality from a truncated network is limited and
the layer split is a per-model tuning problem. Lookahead decoding takes a
different route: it uses Jacobi iteration over a window of future positions
in parallel with the normal decode, harvesting any positions that converge
early, and needs no drafter or training at all. Its gains (about 1.3x to 1.5x
in the paper) are smaller but it composes with other methods and never hurts
correctness, since verification is exact by construction.

## Training the drafter

Drafter quality is a training-data question. A draft model or head trained
on generic text underperforms on the deployment distribution; the strong
results come from training on the target model's own outputs on
representative prompts (self-distillation), so the drafter learns the
target's habits, including its formatting tics and failure modes. EAGLE-style
feature drafters additionally need the target's hidden states, so training
requires a forward pass of the target over the corpus, a real but one-time
cost. Retraining is needed when the target is updated; a stale drafter
against a fine-tuned target loses acceptance measurably, which is an
operational argument for the smallest drafter that meets the acceptance
target.

## Adaptive speculation depth

Fixed gamma is wrong twice: easy stretches (boilerplate, code syntax) accept
8 or more tokens, hard stretches accept 0 or 1. Adaptive schemes set gamma
per step from the drafter's confidence (stop drafting when the drafter's top
probability drops below a threshold) or from recent acceptance history.
Reported gains over the best fixed gamma are on the order of 10 to 20
percent of end-to-end speed, and the mechanism also protects large-batch
operation: under load, confidence-based drafting naturally proposes less,
reducing wasted verification FLOPs exactly when the system can least afford
them. Serving engines expose max speculation depth and increasingly a
target acceptance threshold rather than a single fixed gamma.

## Speculation in disaggregated and batched serving

Speculative decoding assumes spare compute to verify guesses; in a
saturated fleet there is none, and verification tokens displace real tokens.
This is why several production stacks enable speculation only below a batch
threshold or for latency-tier traffic, and disable it for throughput-tier
traffic on the same fleet. The decision variable is the marginal cost of a
verification FLOP at current load, which the scheduler knows and a static
config does not. Any serving evaluation of speculative decoding should
therefore report the load at which it was measured and the batch-size cutoff
policy; a 2.5x speedup at batch 1 and a slowdown at batch 64 are both true
statements about the same system.

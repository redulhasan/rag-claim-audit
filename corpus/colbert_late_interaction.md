# ColBERT and late interaction retrieval

## The MaxSim operator

ColBERT keeps one embedding per token instead of one per document. At query
time it scores a document as the sum over query tokens of the maximum cosine
similarity between that query token's embedding and any document token
embedding: score = sum_i max_j (q_i . d_j). This late interaction preserves
term-level matching (a rare query term can find its exact counterpart token)
while still using contextual embeddings, which is why ColBERT behaves like a
learned, soft version of BM25's term matching and beats single-vector
bi-encoders on most BEIR-style benchmarks by several points of NDCG.

## Storage math

The cost is storage: a document of 300 tokens at 128 dimensions in fp16 is
300 * 128 * 2 = 76.8 KB, versus 1 KB for a single 512-dim document vector.
A 10M-passage index is therefore about 768 GB uncompressed, which is not
shippable. ColBERTv2 compresses with residual quantization: token vectors are
clustered (centroids), and each vector is stored as a centroid id plus a
quantized residual at 1 to 2 bits per dimension, bringing storage to roughly
16 to 32 bytes per token (about 6x to 10x compression) with small quality
loss. PLAID adds centroid interaction pruning and decompression-free scoring
to make search over the compressed index fast enough for production
latencies (tens of ms per query at million scale).

## Two-stage usage in practice

Most deployments do not search the full token index directly. The common
pattern: a fast first stage (BM25 or a bi-encoder) retrieves a few hundred
candidates, and ColBERT re-ranks them with exact MaxSim over stored token
embeddings. This gets most of the quality at a fraction of the latency,
because MaxSim over 300 candidates is cheap. Used as a pure retriever with
PLAID, ColBERT is competitive in latency with dense search at moderate
scale; at very large scale the operational complexity of the token index is
why many teams keep it in the reranking slot.

## MUVERA and fixed-dimensional encodings

MUVERA (2024) converts multi-vector representations into a single
fixed-dimensional vector per document (via fixed-dimensional encodings of
the token set) so standard single-vector ANN indexes can approximate MaxSim
retrieval. It reports recovering most of ColBERT's quality with the serving
simplicity of dense retrieval. The general lesson: late interaction is a
scoring function, and the field keeps inventing index structures to serve it
cheaply, because the quality advantage over single vectors is consistent
while the storage disadvantage is engineering, not mathematics.

## When late interaction earns its cost

The advantage concentrates where term identity matters: entity-heavy
questions, code search, product attributes, and any corpus where a single
rare token decides relevance. On paraphrase-heavy semantic search the gap
over a good bi-encoder narrows to noise. Decide by measurement on your own
queries: run the bi-encoder and ColBERT over the same candidate sets and
compare NDCG on judged queries. If the delta is under a point, the token
index is not worth operating; if it is 3 or more points, it usually is.

## Query-side details

ColBERT pads queries to a fixed length (32 tokens in the original) with
mask tokens that act as learned expansion slots: the extra query vectors
attend softly to document terms beyond the literal query tokens, giving a
controlled form of query expansion inside the scoring function. Removing
them costs measurable quality, which surprises people who expect the query
tokens alone to carry the semantics. Document tokens are filtered (punctuation
embeddings dropped) before indexing, a small storage saving that also
removes noise matches. These details are why reimplementing MaxSim from the
formula alone underperforms the reference system by a visible margin.

## Index build cost

Building a ColBERT index costs one encoder pass per document token plus the
clustering and residual quantization training over a sample of token
vectors. For a 10M-passage corpus this is hours of GPU time and a
multi-hundred-GB intermediate before compression, versus minutes for BM25.
Re-indexing on model change is therefore a planned event, not an
afterthought, and it is one of the practical reasons teams prototype with
ColBERT in the reranking slot (where only candidate documents need token
embeddings, computable on the fly or cached per document) before committing
to a full late-interaction index.

## ColBERT versus cross-encoder reranking

Both are second-stage scorers in many stacks, and they trade differently: a
cross-encoder reads the query and document jointly with full attention and
is the most accurate per candidate, at the highest per-candidate cost
(a full transformer pass per pair). ColBERT precomputes document token
vectors, so its per-candidate cost is a matrix multiply against stored
embeddings, roughly 10x to 100x cheaper per candidate at inference, with
accuracy between a bi-encoder and a cross-encoder. The standard stack uses
each where its cost profile fits: bi-encoder or BM25 to recall hundreds,
ColBERT to order tens, cross-encoder to finalize the top few when the
latency budget allows all three stages.

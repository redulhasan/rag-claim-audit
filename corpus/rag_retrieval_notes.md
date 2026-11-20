# Retrieval for RAG: BM25, dense search, and reranking

## BM25

BM25 scores a document for a query with term frequency saturation and
inverse document frequency:

score = sum over query terms of IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

Typical defaults are k1 = 1.5 and b = 0.75. b controls length normalization:
b = 0 means long documents are not penalized at all. BM25 is exact-match at
heart, so it is very good when the query contains rare terms, acronyms, or
error codes that appear verbatim in the right document, and weak when the
query paraphrases the document in different words.

## Dense bi-encoders

A bi-encoder embeds query and document separately and compares them with
cosine similarity. Because the embeddings are computed independently, all
document vectors can be precomputed and indexed (FAISS, for example), and a
query costs one forward pass plus an nearest-neighbour search. The weakness
is the single-vector bottleneck: one 384-dimensional vector for
all-MiniLM-L6-v2 has to carry the whole meaning of a chunk, and fine
distinctions (negations, swapped numbers, similar entities) often get
smoothed away.

## Reciprocal Rank Fusion

RRF merges ranked lists without needing comparable scores:

RRF_score(d) = sum over lists of 1 / (k + rank(d)), with k = 60 by convention.

The constant is not very sensitive in practice; k = 60 comes from the
original Cormack et al. paper and dampens the weight of top ranks so a
document ranked moderately by both retrievers can beat a document ranked
first by one and nowhere by the other. RRF needs no score calibration
between BM25 (unbounded) and cosine similarity (bounded), which is the main
reason people use it instead of a weighted sum.

## Cross-encoder reranking

A cross-encoder takes the query and one candidate document together as a
single input and outputs one relevance score. It cannot be precomputed, so
it only ever runs on the top candidates from a cheaper first stage, usually
10 to 100 documents. Quality is better than a bi-encoder because attention
runs across the query and document jointly, but cost is one transformer
forward pass per candidate, so latency grows linearly with the rerank depth.
On CPU, reranking 20 candidates with a small model like
ms-marco-MiniLM-L-6-v2 is noticeable: tens of milliseconds per pair adds up
to hundreds of milliseconds per query.

## Chunking and lost in the middle

Header-aware chunking splits on the document structure first (markdown
headers, sections) and only then on size, so a chunk keeps the section it
came from as metadata. Fixed-size chunking is simpler and splits mid-argument
regularly; in these notes it also splits formulas from the sentence that
explains them.

Lost in the middle: language models use information at the start and end of
a long context more reliably than information buried in the middle. For
retrieval this means the rank order you hand the model matters, not just the
set of chunks. It is an argument for reranking and for keeping the final
context short, not for stuffing the top 50 chunks into the prompt.

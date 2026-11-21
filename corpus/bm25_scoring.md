# BM25 scoring in depth

## The formula

BM25 scores a document d for query terms q_i as a sum over terms of
IDF(q_i) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl)), where tf
is the term frequency in d, |d| is document length, and avgdl is the average
document length. IDF is usually log(1 + (N - n + 0.5) / (n + 0.5)) for n
documents containing the term out of N. The two knobs are k1 (term frequency
saturation) and b (length normalization), with defaults k1 = 1.5 (valid
range roughly 1.2 to 2.0) and b = 0.75 inherited from the Okapi system and
baked into Lucene, Elasticsearch, and most libraries.

## What k1 actually does

The tf component saturates: as tf grows, the term's contribution approaches
(k1 + 1) times its IDF, so the 50th occurrence of a word adds almost nothing
over the 10th. At k1 = 1.5, going from tf 1 to 2 raises the factor from 1.0
to about 1.43 (for an average-length document), and from tf 10 to 20 only
from about 2.17 to 2.32. Low k1 (near 0.5) makes BM25 behave almost like a
binary presence score, useful when term repetition is spammy; high k1 (2.0)
rewards genuinely repetitive documents more. Title and header fields exploit
this by repeating terms through field weighting rather than by raising k1.

## What b actually does

b interpolates between no length normalization (b = 0: the denominator
ignores |d|) and full normalization (b = 1). Long documents contain more
terms by chance, so without normalization BM25 systematically favors them;
with full normalization it can over-penalize documents that are long because
they are thorough. b = 0.75 is the compromise that survived decades of TREC
evaluations. Setting b = 0 is right only when document length carries no
information about relevance, for example fixed-size chunks of equal length,
where normalization divides by a near-constant anyway. On variable-length
collections, b = 0 measurably hurts: long documents crowd the top ranks on
common terms.

## IDF behavior and negative values

The classic Robertson IDF, log((N - n + 0.5) / (n + 0.5)), goes negative for
terms appearing in more than half the collection; Lucene's variant adds 1
inside the log to keep it non-negative. On small corpora IDF is coarse: with
29 chunks, a term in 3 chunks and a term in 4 chunks get very different
weights from very little evidence, which is one reason BM25 rankings on tiny
indexes are unstable. Stopword removal interacts here: removing "the" is
uncontroversial, but aggressive stopword lists remove terms like "not" that
flip meaning, and several production bugs trace to exactly that.

## BM25 variants worth knowing

BM25+ adds a small constant delta (typically 1.0) to the tf component so
very long documents are not under-scored for matching terms once each, fixing
a known bias of the original formula on long documents. BM25L applies a
similar correction inside the saturation function. BM25F extends scoring to
weighted fields (title matches count more than body matches). In the other
direction, learned sparse models like SPLADE keep the inverted-index
machinery but replace the tf/IDF formula with learned term weights and
expansions from a transformer; they beat BM25 on benchmarks like BEIR while
keeping its latency profile, at the cost of an encoder at index and query
time.

## Tuning on your own corpus

The defaults are strong; tuning k1 and b on a handful of queries mostly fits
noise. A defensible tuning process needs at least a few hundred judged
queries, a grid over k1 in [0.8, 2.0] and b in [0.3, 0.9], and evaluation by
NDCG or MRR with the differences checked against per-query variance. Reported
gains from careful tuning on TREC-style collections are typically 1 to 3
points of NDCG; gains from fixing tokenization (handling code identifiers,
hyphens, version numbers) are often larger, which is where engineering time
should go first on technical corpora.

## Tokenization decides more than parameters

BM25 sees tokens, and the tokenizer defines what can match. Technical text
is full of identifiers that naive tokenizers destroy: "gpt-4o" split into
"gpt" and "4o", version strings like "2.1.3" shattered into digits, snake_case
and camelCase identifiers split or not depending on the analyzer. A
tokenizer that keeps identifiers whole and also indexes their parts (so
"flash_attention" matches both the full token and "flash" and "attention")
typically gains more on technical corpora than any k1 or b tuning. Case
folding, stemming, and stopword policy are likewise corpus decisions:
stemming helps prose and hurts code search, where "training" and "trainer"
are different symbols.

## Fielded retrieval

When documents have structure, field weighting (BM25F or a simple weighted
sum of per-field BM25 scores) exploits it: a match in a title or header is
worth several matches in body text. RAG indexes over markdown get a cheap
version of this by indexing the header path as a separate field, or by
repeating the section title into the chunk text, which is what several
pipelines do in practice. The weights are hyperparameters like any other
(title weight 2 to 5 is the usual range) and deserve the same judged-query
evaluation rather than intuition.

## BM25 inside hybrid systems

In a hybrid system BM25's job changes: it no longer needs to be right alone,
it needs to be wrong differently from the dense retriever. Its exact-match
strength covers the dense model's weakness on rare tokens, identifiers, and
numbers, and rank fusion handles the rest. Two practical consequences: the
BM25 index should use the same chunking as the dense index so fusion
compares like with like, and BM25 parameters matter less in hybrid than
standalone because fusion is forgiving of either side's mid-rank noise.
What still matters is the candidate pool depth: fusion cannot recover a
document neither retriever placed in the pool, so pools of 50 to 100 per
side are common even when the final top-k is 5.

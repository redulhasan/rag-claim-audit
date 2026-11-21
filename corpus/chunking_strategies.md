# Chunking strategies for retrieval

## The chunking trade in one paragraph

A chunk must be small enough that its embedding is about one thing, and large
enough to contain the answer with its context. Too small (under about 100
tokens) and embeddings match fragments that lack the qualifying sentence; too
large (above about 1000 tokens) and one embedding averages several topics, so
the chunk ranks for everything and answers nothing precisely, and the
generator gets diluted context. Most measured systems land at 200 to 500
tokens with 10 to 20 percent overlap for prose, but the optimum is a function
of the query type: factoid queries want small chunks, synthesis queries want
large ones. Anyone reporting a single best chunk size without the query mix
is reporting an artifact of their eval set.

## Fixed and recursive splitting

Fixed-size splitting cuts every N characters or tokens with overlap; it is
predictable and mixes topics at boundaries. Recursive character splitting
tries separators in order (paragraph breaks, line breaks, sentences, words)
and only cuts mid-sentence when a segment still exceeds the size cap, which
keeps most chunks semantically whole at the same average size. In the
ablation in this repository, header-aware splitting and fixed splitting at
800 characters scored identically on Hit@3 and MRR on a 6-note corpus; the
header-aware version was kept because its citations name a section. That null
result is typical for small clean corpora and should not be extrapolated to
messy document collections.

## Structure-aware splitting

Splitting on the document's own structure (markdown headers, HTML sections,
code function boundaries, legal clauses) keeps units that authors already
made coherent. The metadata is as valuable as the boundary: storing the
header path with each chunk enables citation, filtering, and the
context-prepending trick described in the contextual retrieval note. The
failure mode is variance: real documents have sections from 50 to 5000
tokens, so structure-aware splitting needs a size cap (recursive split inside
oversized sections) and a merge rule for tiny sections, or the index ends up
with a few giant chunks and many stubs.

## Semantic chunking

Semantic chunking embeds sentences sequentially and cuts where the cosine
similarity between adjacent sentence (or sentence-group) embeddings drops
below a threshold, commonly a percentile of the document's own similarity
distribution (the 95th percentile breakpoint in the popular LangChain
implementation) rather than an absolute value. It finds topic boundaries
that structure misses, in unstructured text. Costs: an embedding pass over
every sentence before the real indexing pass, threshold sensitivity across
domains, and boundaries that are hard to explain or reproduce when the
embedding model changes. On technical documentation with real headers it
rarely beats structure-aware splitting; on transcripts and prose without
structure it is often the best available option.

## Late chunking

Late chunking inverts the order: embed the whole document with a long-context
embedding model, then pool the token embeddings within each chunk boundary
to get chunk vectors. Each chunk vector is therefore informed by the full
document context (a pronoun in chunk 12 is embedded with its antecedent in
chunk 3 present), while the chunks themselves stay small and precise.
Boundaries can be naive fixed splits because the context problem is handled
by the embedding, not the boundary. Requirements: an embedding model with
real long-context quality (8k tokens or more) and mean-pooling access to
token-level embeddings, which not all embedding APIs expose. Reported gains
are largest on documents with heavy cross-references.

## Propositional and small-to-big retrieval

Propositional chunking rewrites the document into atomic self-contained
statements (with an LLM) and indexes those; retrieval precision rises
because each unit is one fact, and the parent passage is fetched for
generation. Small-to-big (parent-document) retrieval is the same shape
without rewriting: index small chunks for matching, return their parent
section to the generator. Both trade index size and an offline LLM bill
(propositional) or recall granularity (parent) for answer quality. The
evaluation trap: Hit@K measured at chunk level is not comparable across
these strategies, because the retrieved units differ; measure at the level
of what reaches the generator.

## Overlap, honestly measured

Overlap duplicates boundary content into neighboring chunks so a fact
straddling a cut survives in at least one chunk. It costs index size
linearly (20 percent overlap is 20 percent more vectors) and can hurt
precision by making near-duplicate chunks crowd the top-k. In controlled
tests on clean corpora, overlap between 0 and 20 percent usually changes
Hit@K by less than the noise between query sets; its real value is insurance
on messy documents where sentence boundaries are unreliable. Measure it on
your corpus or leave the default at 10 to 15 percent and spend the effort on
boundaries instead.

## Chunking code, tables, and other non-prose

Prose assumptions fail on structured content. Code should be chunked on
syntax boundaries (functions, classes) with the signature kept with the
body; splitting mid-function produces chunks that match queries about the
function name while lacking its logic. Tables need their headers repeated
into every row chunk or the numbers are meaningless; a row chunk without
column names retrieves well for nothing. Mixed documents benefit from
per-type strategies routed by a cheap classifier, at the cost of pipeline
complexity. The evaluation sets for such corpora must include queries whose
answers live in the structured parts, or the chunking choice for them is
never actually measured.

## Multilingual and noisy text

Sentence splitters and stopword assumptions are language-specific: several
languages do not mark sentences with periods, and OCR output breaks the
punctuation that recursive splitters rely on. For noisy corpora, character
and token windows with generous overlap are more forgiving than
structure-based methods whose structure detection fails silently. Embedding
quality also varies by language far more than chunking quality does; on a
multilingual corpus, upgrading the embedding model usually beats every
chunking change combined. Measure per language slice, because aggregate
metrics hide a broken language behind the majority one.

## Evaluating a chunking change

Chunking changes are retrieval changes and should be evaluated like them:
fixed query set with judged relevance at the passage level, Hit@K and MRR
reported per query with the deltas listed, and the index rebuilt from
scratch for both variants (chunking interacts with IDF statistics, so
reusing corpus statistics across variants contaminates the comparison). The
result to expect on a clean corpus is often "no significant difference", as
in this repository's own ablation; that is a useful result, because it
redirects effort to the parts of the pipeline where differences exist. A
chunking change adopted without this evaluation is a superstition with a
config file.

# Contextual retrieval

## The failure it addresses

A chunk that says "Revenue grew 3 percent over the previous quarter" is
unretrievable by the query "How much did Acme's revenue grow in Q2 2023"
because the chunk never names the company or the quarter: the context lived
in the surrounding document and was cut away by chunking. This is the most
common silent failure in naive RAG pipelines over reports, manuals, and
notes, and no embedding model fixes it, because the information is absent
from the indexed text, not badly encoded.

## The method

Contextual retrieval (Anthropic, 2024) prepends a short generated context to
each chunk before embedding and before BM25 indexing: an LLM reads the whole
source document plus the chunk and writes 50 to 100 tokens situating it
("This chunk is from Acme's Q2 2023 earnings report, in the revenue
section"). The augmented text is what gets embedded and keyword-indexed; the
original chunk is what gets shown and cited. Two variants follow: contextual
embeddings alone, and contextual embeddings plus contextual BM25.

## The published numbers

On Anthropic's evaluation sets (code, fiction, papers, and support docs),
top-20 retrieval failure rate fell by 35 percent with contextual embeddings
alone (from a 5.7 percent baseline failure rate to 3.7 percent), by 49
percent when contextual BM25 was added (to 2.9 percent), and by 67 percent
when the pipeline also reranked (to 1.9 percent). The numbers are averages
across their domains and their chunking setup; the ordering of the three
steps (embeddings, then BM25, then reranking) is the part that transfers,
not the exact percentages. Note the reranking step contributed as much as
the contextualization itself in the combined system.

## Cost and the prompt caching trick

Generating a context blurb per chunk costs one LLM call per chunk, which at
10k chunks and standard API prices would be prohibitive without caching.
The trick: the full document is the shared prefix of every chunk's prompt,
so with prompt caching the document is paid for once at the write rate and
read at the cached rate (about 10 percent of base input price on Anthropic's
pricing) for the remaining chunks. Anthropic's worked example puts the
one-time cost at about 1.02 USD per million document tokens. The index-time
bill is therefore acceptable for static corpora and questionable for corpora
that change hourly.

## When it does not pay

If chunks are already self-contained (FAQ entries, API documentation with
repeated context, these notes with explicit section headers), the generated
context adds little signal and can add noise: a wrong context blurb actively
misleads retrieval. Fast-changing corpora pay the generation cost repeatedly.
And the method does nothing for the other big failure class, questions whose
answers span several chunks; that needs query decomposition or parent
retrieval. Measure the failure class before buying the fix: sample 50 failed
retrievals and count how many failed for missing context specifically.

## Interaction with citations

Because the context is prepended for indexing only, citations still point at
the original chunk and the generator sees the original text (optionally with
the context shown as metadata). Mixing this up, showing users generated
context as if it were source text, is both a correctness and a trust bug.
Keep three versions straight in the data model: original text (display and
citation), augmented text (indexing), and section metadata (filtering).

## Generating contexts at scale

The generation prompt matters: the model is told to give the chunk a short
succinct context explaining what the chunk is about within the whole
document, and outputs are constrained to 50 to 100 tokens. Quality control
is light but real: contexts that merely repeat the chunk's first sentence
add nothing, and contexts that hallucinate a wrong document identity are
actively harmful, so a sample audit of generated contexts belongs in the
pipeline's first run. For corpora in the millions of chunks, a small fast
model is the only economically sane generator, and the quality difference
between a small and a large generator for this narrow task is minor in
reported practice.

## How much context is enough

The context needs to carry the identifiers the chunk omits: document title,
date or version, section topic, entity names, and the units or scope of any
numbers. It does not need to summarize the chunk's content, which the chunk
already has. Fifty to one hundred tokens covers this for most documents;
longer contexts start duplicating content and dilute the embedding with
generic text. A cheap ablation decides it on your corpus: generate contexts
at two lengths, index both, and compare retrieval failure on the judged set.
Skipping this and copying a number from a blog post is how pipelines end up
with 300-token contexts that cost more and retrieve worse.

## Failure modes of the fix itself

Generated context can be wrong in ways the original text never was: the
generator misattributes a chunk to the wrong product version, merges two
entities, or states a date the document never gave. Because the context is
prepended to the indexed text, these errors are searchable and will be
retrieved confidently. The mitigations are to keep the context extractive in
spirit (instruct the generator to use only document metadata and headers),
to log the context with the chunk for auditability, and to remember that the
generator never sees the generator's context: if a user asks why a chunk was
retrieved, the honest answer includes showing them the prepended text.
Systems that hide it cannot debug their own retrieval.

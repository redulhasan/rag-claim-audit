# KV cache serving: paging, prefix caching, and cache quantization

## PagedAttention in depth

PagedAttention divides the KV cache into fixed-size blocks (16 tokens per
block in vLLM's default) and keeps a block table per sequence mapping logical
block positions to physical blocks anywhere in GPU memory, exactly like an
operating system's page table. Allocation happens block by block as tokens
are generated, so a sequence only holds the blocks it has filled. Internal
fragmentation is at most one partial block per sequence (15 tokens worst
case), versus the 60 to 80 percent waste of contiguous pre-allocation at max
length, and measured total waste falls under 4 percent. The freed memory is
spent on more concurrent sequences, which is where the throughput gains come
from; single-sequence latency is roughly unchanged.

## Prefix caching

Many requests share prefixes: the same system prompt, few-shot examples, or
document context. Prefix caching hashes each full block's token ids (plus the
hash of its prefix, so identical text at different positions does not match)
and reuses the physical block across requests. A radix-tree variant
(SGLang's RadixAttention) keeps the cache as a tree of token sequences with
LRU eviction at the leaves. The payoff is time-to-first-token: a request
whose 2000-token prompt is fully cached skips the entire prefill, and
prefill is the expensive part (2000 tokens of compute versus 2000 cache
reads). Hit rates in multi-tenant chat workloads are high because system
prompts repeat; in single-user workloads with unique documents they are near
zero, so the feature should be evaluated on the actual traffic mix.

## Chunked prefill

A long prefill monopolizes the GPU and stalls every decode in a continuous
batching system, spiking time-between-tokens for all running requests.
Chunked prefill splits a long prompt into chunks (for example 512 or 1024
tokens) and schedules one chunk per step alongside the decodes. This bounds
the per-step compute, smoothing TPOT at the cost of slightly worse prefill
efficiency (smaller matmuls) and a modest TTFT increase for the long request
itself. vLLM and TensorRT-LLM both expose the chunk size as a tuning knob;
the right value depends on the SLO split between TTFT and TPOT.

## KV cache quantization

The cache itself can be quantized. FP8 cache storage halves cache bytes with
small quality cost and is supported in vLLM and TensorRT-LLM; per-token
scales keep it accurate. INT4 cache quantization is harder because keys have
outlier channels: KIVI's finding is that keys should be quantized per-channel
and values per-token, reflecting their different outlier structure, and that
a small residual window of recent tokens kept in full precision protects
quality. Cache quantization multiplies the effective capacity of the same
memory by 2x (FP8) to about 4x (INT4 with residuals), and at long context the
cache, not the weights, is what runs out, so this is often worth more than
weight quantization for serving.

## Eviction and preemption

When memory runs out mid-generation, serving systems either preempt (drop a
sequence's cache and recompute later, or swap it to CPU memory) or refuse
new requests. Recompute preemption is simple and wasteful; swapping costs
PCIe bandwidth but preserves the work. Priority matters: preempting the
newest, shortest sequence loses the least completed work. Under sustained
overload the correct behavior is admission control (queue or reject at the
door) rather than letting every request degrade, because a system at 99
percent cache occupancy with constant preemption delivers worse goodput than
one serving fewer requests cleanly.

## Cache-aware scheduling

Once prefix caching exists, routing becomes a cache problem: sending a
request to a replica that already holds its prefix can cut TTFT by more than
any kernel optimization. Load balancers for LLM serving therefore track
approximate prefix state per replica (or use consistent hashing on the
prefix) instead of round-robin. The same logic applies within one engine:
scheduling requests that share prefixes back-to-back keeps the shared blocks
hot. These scheduling gains are workload-dependent in the extreme, which is
why serving benchmarks that use unique random prompts systematically
underestimate production TTFT performance.

## Block size selection

Block size trades fragmentation against block-table overhead and prefix
matching granularity. Smaller blocks (8 tokens) waste less on partial
blocks and let prefix caching match shorter shared prefixes, but double the
block table size and the number of blocks the attention kernel must walk
versus 16. Larger blocks (32 or 64) do the reverse and also coarsen prefix
sharing: a shared 100-token prefix matches only its first 96 tokens at
block size 32. The vLLM default of 16 is a reasonable middle; workloads
dominated by very long shared system prompts sometimes prefer larger blocks
because the shared region is many whole blocks either way, while workloads
with many short unique requests prefer smaller. As with most serving knobs,
the default is fine until a profile says otherwise.

## Cache state across replicas

Prefix caches are per-engine and per-replica; nothing shares KV blocks
across machines in mainstream stacks, because moving a block costs as much
bandwidth as recomputing small prefixes and the coherence bookkeeping is
unpleasant. This makes replica warm-up a real effect: a freshly scaled-out
replica has an empty prefix cache and serves its first requests at full
prefill cost, which matters during traffic spikes exactly when scale-out
happens. Mitigations are boring and effective: warm replicas with synthetic
traffic carrying the common system prompts before joining the pool, and
drain replicas slowly so their caches die with their last requests rather
than all at once.

## Multi-modal and hybrid caches

Vision-language models add image-token KV entries that are large, shared
across questions about the same image, and worth caching by image hash
rather than by token prefix alone; several engines special-case this.
Hybrid models (attention layers interleaved with state-space or recurrent
layers) maintain two kinds of state with different paging behavior, and the
recurrent state cannot be block-paged the same way, which complicates
preemption: dropping a sequence means dropping both states consistently.
These are the current edge cases where serving abstractions leak, and they
are worth asking about explicitly when evaluating an engine for a non-pure-
transformer model.

# Attention variants: MHA, MQA, GQA, and sliding window

## Multi-head attention cost accounting

In multi-head attention (MHA) with h heads and head dimension d_h, every head
keeps its own key and value projections, so the KV cache per token is
2 * n_layers * h * d_h values. For a 32-layer model with h = 32 and d_h = 128
in fp16 that is the familiar 0.5 MB per token. The compute per decode step is
small next to the memory traffic: each new token reads the entire cache once,
so decode throughput is set by cache bytes divided by HBM bandwidth. Any
variant that shrinks the number of KV heads divides both cache size and decode
bandwidth by the same factor.

## Multi-query attention

Multi-query attention (MQA) shares a single K head and a single V head across
all query heads. The KV cache and the decode bandwidth drop by a factor of h
(32x in the example above, from 0.5 MB to about 16 KB per token). The cost is
model quality: one shared key space has to serve every query head, and the
original MQA paper and follow-ups report a small but real quality drop on
reasoning-heavy evaluations. MQA also makes training slightly awkward because
the shared K, V receive summed gradients from all heads. It is used where
decode bandwidth is the binding constraint, for example in some PaLM and
Falcon configurations.

## Grouped-query attention

Grouped-query attention (GQA) interpolates between MHA and MQA: the h query
heads are split into g groups, and each group shares one KV head. With h = 32
and g = 8, the cache shrinks 4x (0.5 MB to 0.125 MB per token in the example)
while keeping 8 distinct key spaces. Llama 2 70B uses GQA with 8 KV heads for
64 query heads, an 8x cache reduction. Empirically GQA at g = 8 matches MHA
quality within noise on most benchmarks while MQA does not, which is why GQA
became the default in open models after 2023. An existing MHA checkpoint can
be converted by mean-pooling the K and V projections inside each group and
fine-tuning briefly (about 5 percent of the original pretraining compute in
the GQA paper).

## Sliding window attention

Sliding window attention (SWA) restricts each token to the previous W tokens;
Mistral 7B uses W = 4096. Stacked over L layers the effective receptive field
is about L * W tokens, because information moves one window per layer, so a
32-layer model still mixes information across roughly 131k positions in
principle, though the mixing is lossy compared to full attention. The KV cache
becomes a rolling buffer of the last W tokens per layer, so cache memory stops
growing with sequence length: at W = 4096 the cache is capped at the 4096-token
cost no matter how long the stream runs. The failure mode is exact recall of a
fact placed more than W tokens back in a single layer's view, which is why
some models alternate SWA layers with full-attention layers.

## Attention sinks and streaming

StreamingLLM showed that keeping the first few tokens of the sequence (the
attention sink, often 4 tokens) plus a rolling window preserves perplexity far
better than a rolling window alone, because softmax forces attention mass
somewhere and early tokens absorb it. Removing the sink tokens from a windowed
cache causes a sharp perplexity spike even though those tokens carry little
content. Practical serving stacks therefore pin the sink tokens and evict from
the middle of the window, and the same observation explains why naive cache
eviction policies that drop the oldest tokens fail.

## Sparse and hybrid patterns

Block-sparse and strided patterns cut the N^2 term by attending to a subset of
positions, but on modern GPUs dense tiled kernels are so much faster per FLOP
that sparse patterns usually need very high sparsity (above 90 percent) to win
in wall clock. The pattern that survived in production is the hybrid: most
layers local (sliding window), a few layers global. Gemma 2 alternates local
(4096 window) and global layers 1:1; Gemma 3 uses a 5:1 local-to-global ratio
with a 1024-token local window. Hybrids keep most of the cache and bandwidth
savings of SWA while retaining a few layers that can retrieve from the full
context.

## Choosing a variant

The decision is mostly a serving budget calculation. Compute the KV cache per
token for the candidate variant, multiply by target context length and
concurrent sequences, and check it against the memory left after weights. If
decode is the bottleneck and quality budget is tight, GQA with 8 groups is the
usual answer. If context is effectively unbounded (streaming), a windowed
variant with sink tokens is the only one whose memory stays flat. Retrofitting
a variant onto a trained checkpoint costs fine-tuning; choosing it before
pretraining costs nothing, which is why new models rarely ship plain MHA above
about 13B parameters.

## Interaction with KV cache compression

Variant choice and cache quantization multiply. GQA with 8 KV heads cuts the
cache 8x; storing that cache in FP8 cuts it another 2x; the two compose to
16x against an fp16 MHA baseline, taking the 0.5 MB per token example to
about 31 KB per token. The errors also compose, though: fewer KV heads means
each stored head carries more information, and quantization error per head
has fewer siblings to average against. Published serving results suggest the
combination is safe at FP8 and needs measurement at INT4. When capacity
planning, compute the product of the factors first and then validate quality
at that operating point, rather than validating each optimization in
isolation and assuming they compose.

## Training-side differences

The variants are not free at training time. MQA and GQA reduce the K and V
projection sizes, so they slightly reduce parameters and projection FLOPs,
but the attention computation itself (scores and weighted sums) is unchanged
per query head. Gradient flow differs: a shared KV head receives the summed
gradient from all query heads in its group, which changes the effective
learning dynamics of the K and V projections and is one reason converted
checkpoints (mean-pooled from MHA) need a short fine-tune to recover. Group
size also interacts with tensor parallelism: KV heads must divide across the
TP degree or be replicated, so GQA with 8 KV heads on a TP-8 setup puts
exactly one KV head per device, and TP degrees above the KV head count force
replication that quietly returns some of the memory the variant saved.

## Utilization notes

At decode time the win from fewer KV heads is a bandwidth win, and it shows
up only when decode is actually bandwidth bound on the cache read. At large
batch sizes the weight read amortizes and compute grows, so the relative
gain from GQA shrinks; at batch 1 and long context it is close to the full
head-count ratio. Prefill is compute bound and barely notices the variant,
except that a smaller cache means more room for concurrent sequences, which
raises throughput indirectly. Benchmarks that report a single speedup number
for GQA without batch size and context length are withholding the variables
that decide the answer.

# Transformers and attention

## Scaled dot-product attention

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. The 1/sqrt(d_k) term is not
decoration: for large d_k the dot products grow in magnitude, the softmax
saturates, and gradients through it get very small. With d_k = 64 the scale
factor is 1/8.

Each head uses its own projections with d_k = d_model / h. A model with
d_model = 512 and h = 8 heads therefore runs 8 attention operations in
parallel at d_k = 64, then concatenates and projects the result.

Causal decoding applies a mask before the softmax so position i can only
attend to positions <= i. Without the mask the model would cheat during
training by looking at the token it is supposed to predict.

## Complexity

Self-attention is O(n^2 * d) in sequence length n because every position
scores against every other position. The feed-forward network is O(n * d^2).
For short sequences the feed-forward cost dominates; for long contexts the
quadratic attention term takes over, which is why most long-context work
attacks the attention term first.

## KV cache

During autoregressive decoding, keys and values for past tokens do not
change, so they are cached instead of recomputed. Per token, the cache holds
2 * n_layers * d_model values (one K and one V per layer). For a 32-layer
model with d_model = 4096 in fp16, that is 2 * 32 * 4096 * 2 bytes = 0.5 MB
per token, or roughly 500 MB for a 1000-token context, per sequence.

The trade is memory for compute: decoding step t costs O(t * d) attention
work instead of O(t^2 * d), at the price of the cache above. Batch size in
production serving is usually limited by KV cache memory, not weights.

## RoPE

Rotary position embedding rotates each pair of dimensions of q and k by an
angle proportional to the position. The rotation is applied to the vectors
themselves, not added as a separate embedding, so the dot product between a
query at position m and a key at position n depends on the relative offset
m - n. Base frequency is commonly 10000. RoPE extrapolates poorly beyond the
training context length unless the base is rescaled (NTK-aware scaling or
similar tricks change the base, not the rotation idea itself).

## Multi-head, grouped-query, and multi-query attention

Multi-query attention (MQA) shares one K/V head across all query heads.
Grouped-query attention (GQA) is the middle point: groups of query heads
share one K/V head each. Both shrink the KV cache by the sharing factor,
which matters far more at inference time than the small quality difference
seen in most benchmarks. GQA with 8 KV heads instead of 32 cuts cache size
by 4x for the same d_model.

# FlashAttention and IO-aware attention

## The IO problem in standard attention

Standard attention computes S = Q K^T, applies softmax row by row, and then
multiplies by V. The score matrix S has N x N entries for sequence length N,
and a naive implementation writes S and the probability matrix P to HBM
(high-bandwidth memory) and reads them back for each step. At N = 16384 in
fp16, one score matrix for one head in one layer is 16384^2 * 2 bytes = about
537 MB. GPU SRAM is tiny by comparison (about 192 KB per streaming
multiprocessor on an A100, tens of MB in aggregate) while HBM bandwidth is
about 1.5 to 2 TB/s on an A100 and about 3.35 TB/s on an H100. Attention at
long context is therefore memory-bandwidth bound, not FLOP bound: the tensor
cores sit idle waiting for the N x N matrices to move.

## Tiling and the online softmax

FlashAttention never materializes S or P. It tiles Q into blocks of rows and
K, V into blocks of columns sized to fit SRAM, and computes softmax
incrementally with the online softmax trick: for each block it keeps a running
maximum m and a running sum of exponentials l, rescaling the output
accumulator whenever a new block raises the maximum. The rescaling factor
exp(m_old - m_new) corrects the partial results exactly, so the output is
numerically identical to the unfused computation, not an approximation. Block
sizes are typically 64 to 256 rows, chosen so the Q block, K and V blocks, and
the accumulator fit in SRAM at once.

## Memory footprint and measured speedups

Because only the output (N x d) and the softmax statistics (two vectors of
length N, the max and the log-sum-exp) are stored, attention memory drops from
O(N^2) to O(N). The FlashAttention paper reports 10x to 20x less memory than a
standard PyTorch implementation at sequence lengths of 4k to 16k, and 2x to 4x
wall-clock speedups on GPT-style training. The saved log-sum-exp is also what
the backward pass needs, so the N x N matrix is never stored for the backward
either; it is recomputed block by block from Q, K, V in SRAM.

## Backward pass by recomputation

Recomputing S and P in the backward pass costs extra FLOPs but saves HBM
traffic, and because attention is bandwidth bound the trade is a net win. This
is the same argument as activation checkpointing applied inside one operator:
spend arithmetic, which is cheap and underused, to save memory movement, which
is the bottleneck. The backward kernel recomputes each block of S from the
stored Q, K blocks, rebuilds P from the saved log-sum-exp, and accumulates dQ,
dK, dV without writing any N x N tensor to HBM.

## FlashAttention-2 changes

FlashAttention-2 restructured the kernel rather than the math. It parallelizes
over sequence-length blocks across thread blocks in addition to batch and
heads, which matters when batch * heads is small (long-context training with
batch size 1 or 2 underuses the GPU in version 1). It also swaps the loop
order so each thread block owns a Q block and iterates over K, V blocks,
reducing shared-memory traffic, and it cuts non-matmul FLOPs by deferring the
softmax rescaling. The result is roughly 2x faster than FlashAttention-1,
reaching about 70 percent of the theoretical peak FLOPS on an A100 (about 225
TFLOPS in bf16 for the attention kernel on large models).

## FlashAttention-3 and FP8

FlashAttention-3 targets the H100 specifically: it uses the asynchronous
tensor cores (WGMMA) and TMA data movement, overlaps the softmax computation
with the matrix multiplies (producer-consumer warp specialization), and adds
FP8 support with block scaling and incoherent processing to control outlier
damage. Reported numbers reach about 75 percent of H100 peak in bf16 and up to
about 1.2 PFLOPS in FP8 on attention-heavy shapes. The FP8 path is a training
and inference trade: roughly 2x the bf16 throughput for a small perplexity cost
that has to be measured per model, not assumed away.

## Exactness, dropout, and masking

FlashAttention is an exact attention implementation: with the same inputs and
no dropout it produces the same outputs as the textbook formula up to floating
point reassociation. Dropout inside attention is supported by storing a
Philox RNG seed and offset instead of a mask tensor, so the mask is
regenerated in the backward pass rather than stored; the RNG state is a few
bytes per layer instead of N x N booleans. Causal masking is handled by
skipping blocks that are entirely above the diagonal, which also halves the
work for causal models at long context.

## When FlashAttention does not help

The gains shrink when attention is not the bottleneck. Short sequences (below
about 512 tokens) are dominated by the surrounding MLP and projection work, so
an IO-aware attention kernel changes total step time by only a few percent.
Decode-time inference with batch size 1 is bound by reading the KV cache and
the weights, not by the score matrix, which is why serving stacks pair
FlashAttention-style kernels with paged KV cache management instead of relying
on either alone. Head dimensions above 256 and some sliding-window or
cross-attention shapes also fall back to slower paths in several
implementations, so the kernel actually dispatched is worth checking in a
profile before attributing a speedup to it.

## Numerics of the rescaling

The online softmax maintains, per row block, the running maximum m and the
running sum l of exp(x - m). When a new block's maximum exceeds m, both l and
the output accumulator are multiplied by exp(m_old - m_new), a factor in
(0, 1], so no intermediate value can overflow: every exponential argument is
at most 0 by construction. The subtlety is precision, not range. The
accumulator lives in fp32 in the kernel even when inputs are bf16, because
repeated rescaling of a bf16 accumulator would compound rounding across the
dozens of blocks in a long row. The final division by l happens once, at the
end. This is also why FlashAttention's outputs differ from a naive bf16
implementation in the last bits: the naive version often materializes P in
bf16, while the fused kernel keeps the accumulation in fp32, so the fused
kernel is frequently the more accurate of the two, an unusual case of the
faster implementation also being the more precise one. When debugging a
mismatch between a reference implementation and a fused kernel, check the
accumulation dtype of the reference before blaming the kernel.

## Kernel fusion boundaries

FlashAttention fuses the two matmuls and the softmax, but the surrounding
operations stay outside: the Q, K, V projections, the output projection, any
bias or mask materialization, and the rotary embedding application (which
some kernels do fuse). This matters for profiling: at moderate sequence
lengths the projections are GEMMs with arithmetic intensity high enough to
be compute bound, and attention is only part of the layer time. Amdahl's law
applies directly: if attention is 30 percent of layer time at your sequence
length, an infinitely fast attention kernel caps the gain at about 1.43x.
The reported 2x to 4x figures come from long-sequence regimes where attention
dominates. It also explains why end-to-end training speedups from enabling
FlashAttention on a 2048-token workload are often a modest 10 to 20 percent
even though the kernel itself is several times faster.

## Flash decoding for inference

Training-style FlashAttention parallelizes over query blocks; a decode step
has exactly one query token per sequence, so there is nothing to parallelize
on the query side. Flash-Decoding instead splits the KV cache along the
sequence dimension: each thread block computes a partial attention output
over a slice of the cache with its own running max and sum, and a small
reduction kernel combines the partials using the same log-sum-exp rescaling.
This turns a memory-bound single-row operation into a parallel one and
matters most at long context and small batch, exactly where decode latency
is worst. Split-K factors of 8 to 16 are typical at 32k context. Serving
engines combine this with paged cache layout, so the split boundaries follow
block boundaries in the page table.

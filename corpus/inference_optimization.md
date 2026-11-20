# Inference optimization

## Quantization: INT8 and INT4

Weight-only quantization stores weights in fewer bits and dequantizes on
the fly during the matmul. INT8 halves weight memory versus fp16 with
usually negligible quality loss. INT4 (GPTQ, AWQ) quarters it: a 7B model
goes from 14 GB in fp16 to about 3.5 GB of weights, plus scales and zero
points per group (group size 128 is common), so the real figure is closer
to 4 GB. Quality loss at INT4 is small on perplexity but shows up first on
tasks that need exact arithmetic or rare-token recall.

Activation quantization is harder than weight quantization because
activations have outlier channels with much larger magnitudes. That is why
most open tooling quantizes weights only, and why methods like SmoothQuant
exist specifically to move the difficulty from activations to weights.

## vLLM and PagedAttention

PagedAttention stores the KV cache in fixed-size blocks (16 tokens per
block is the vLLM default) that do not need to be contiguous, borrowing the
paging idea from operating systems. The practical effect: KV cache memory
waste drops from 60 to 80 percent with naive contiguous allocation to under
4 percent, and the freed memory goes to larger batches. vLLM also uses
continuous batching, where sequences join and leave the batch at every
decoding step instead of waiting for the longest sequence in a static batch.

Throughput claims for PagedAttention are about memory efficiency enabling
batch size, not about making a single sequence decode faster. Single-stream
latency is roughly unchanged; tokens per second at fixed hardware with many
concurrent users is what improves.

## Speculative decoding

A small draft model proposes gamma tokens ahead; the large model verifies
them in one forward pass. Accepted tokens are kept, the first rejected
token is resampled from the corrected distribution, and the process repeats.
The output distribution is identical to decoding from the large model alone
when the verification rule is applied correctly - this is a lossless speed
trick, not an approximation.

Speedup depends entirely on the acceptance rate. With gamma = 5 and an
acceptance rate around 0.7, expect roughly 2 to 3x fewer large-model steps.
If the draft model disagrees often (different tokenizer, different domain),
the verification passes get wasted and the speedup can drop below 1x, i.e.
slower than plain decoding.

## Latency terms that get confused

Time to first token (TTFT) is dominated by the prefill pass over the prompt
and grows with prompt length. Inter-token latency (ITL) is the per-step
decode cost and is dominated by memory bandwidth, because each step reads
all weights (and the KV cache) to produce one token per sequence. Prefill is
compute bound, decode is bandwidth bound; optimizing one does not fix the
other, and most "the model is slow" reports never say which of the two the
user actually waited on.

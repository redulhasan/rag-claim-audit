# Mixed precision training: fp16, bf16, and loss scaling

## The two 16-bit formats

fp16 has 1 sign bit, 5 exponent bits, and 10 mantissa bits: range up to
65504, relative precision about 2^-11 (about 0.05 percent). bf16 has 8
exponent bits (the same as fp32) and 7 mantissa bits: range up to about
3.4e38, relative precision about 2^-8 (about 0.4 percent). The trade is
exact: bf16 keeps fp32's dynamic range and gives up precision; fp16 keeps
precision and gives up range. Gradient values in deep networks routinely
fall below 1e-5 and occasionally below fp16's smallest normal (6.1e-5) or
even its smallest subnormal (6e-8), which is the entire reason fp16 needs
loss scaling and bf16 mostly does not.

## Loss scaling for fp16

Loss scaling multiplies the loss by a factor S (so all gradients scale by S)
before the backward pass, shifting gradients up into fp16's representable
range, then divides the gradients by S before the optimizer step. Dynamic
loss scaling adjusts S automatically: start at 2^15, double S after an
interval of steps without overflow (2000 steps is the common growth
interval), and halve S and skip the step whenever an inf or NaN gradient is
detected. A badly tuned static scale either overflows constantly (training
stalls on skipped steps) or underflows the small gradients to zero (the
model silently stops learning its rare features). bf16 training skips this
machinery entirely, which is most of why bf16 won.

## Master weights and accumulation precision

In both formats the optimizer keeps fp32 master weights: the update
lr * grad is often smaller than the bf16 weight's last mantissa bit, so
applying updates directly in bf16 loses them to rounding (a weight of 1.0
with an update of 1e-4 needs relative precision below bf16's 0.4 percent and
would round back to 1.0). The standard layout: forward and backward in bf16,
optimizer state and master weights in fp32, gradients accumulated in fp32 or
bf16 depending on framework. Stochastic rounding is the research alternative
that makes pure-bf16 updates unbiased, used in some low-precision training
experiments but not the mainstream default.

## Gradient clipping

Global-norm clipping rescales all gradients by min(1, max_norm / ||g||) with
max_norm = 1.0 the near-universal default in LLM training. It exists because
loss spikes (a bad batch, a data glitch) can otherwise take a step large
enough to damage the model permanently; with clipping the worst case is one
wasted step. Clipping interacts with loss scaling order (unscale first, then
clip, then step) and with Adam: clipping changes the gradient direction's
magnitude but Adam is largely scale-invariant per coordinate, so clipping's
main effect in Adam training is on the moment estimates during the spike,
which is still worth having. A clip fraction persistently above a few percent
of steps signals a data or learning-rate problem, not a clipping problem.

## Where 16-bit actually breaks

Three recurring failure cases. First, reductions: sums over long sequences
or large batches in bf16 accumulate error; attention softmax denominators
and loss sums are computed in fp32 inside kernels for this reason. Second,
the final LM head and loss: logits in bf16 lose the small differences between
similar tokens, so the loss is computed from fp32-cast logits. Third,
optimizer second moments in bf16: Adam's v estimate spans many orders of
magnitude and is kept in fp32 in every mainstream recipe. Knowing which
tensors a framework keeps in fp32 is part of reading a training config;
"bf16 training" always means a mixed layout, never uniform bf16.

## Speed expectations

On A100 and H100, bf16 tensor-core throughput is 2x fp32 (non-tensor) at the
same shapes, and memory traffic halves, so mixed precision is worth roughly
1.5x to 2x end-to-end on training step time, not the 2x of the peak numbers,
because norms, softmax, optimizer step, and communication stay in higher
precision or stay bandwidth bound. FP8 doubles the tensor-core rate again on
H100, which is why the frontier moved to FP8 recipes with the scaling
machinery described elsewhere in these notes.

## Accumulation inside attention and reductions

Two reductions deserve fp32 even in aggressive bf16 setups. The attention
softmax denominator sums up to sequence-length many exponentials; in bf16
the running sum loses small terms, biasing attention weights at long
context, so kernels accumulate it in fp32. The cross-entropy loss sums over
the batch and the vocabulary-normalization terms; frameworks cast logits to
fp32 before the log-softmax, which at vocabulary 128k and batch tokens 1M
makes the logits tensor itself a memory line item (1M * 128k * 4 bytes is
512 GB if materialized naively, which is why fused chunked cross-entropy
exists). Precision decisions and memory decisions in these two spots are
the same decision.

## Embeddings, norms, and other fp32 holdouts

LayerNorm and RMSNorm compute means and variances over the hidden dimension;
in bf16 the variance estimate is noisy enough to matter, so norms run in
fp32 internally in most kernels and cast at the boundary. Embedding lookups
are memory-bound gathers where precision costs bandwidth, not accuracy, and
embeddings are commonly stored in the compute dtype. Rotary embedding
cosine and sine tables are computed in fp32 and cached; applying them in
bf16 is fine because the rotation is norm-preserving and the error does not
accumulate across layers the way reduction error does. The pattern to
internalize: reductions and normalizations in fp32, elementwise and matmul
in the low precision format.

## A measurement protocol for precision changes

Changing precision settings (enabling FP8, dropping master weights, moving
norms) should be gated on a fixed protocol: same data order, same seed,
short run (a few hundred steps) comparing loss curves against the reference
configuration, with divergence defined in advance (for example final loss
within 0.5 percent). Loss curves are noisy enough that eyeballing two runs
proves nothing; the paired comparison with identical data order removes
most of the noise. Only after the short gate passes does the long run
start. Teams that skip the gate discover precision damage as a mysterious
quality gap weeks later, when the ablation is no longer cheap.

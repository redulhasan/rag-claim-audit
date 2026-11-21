# SmoothQuant and FP8 training

## The activation outlier problem

Weight-activation INT8 quantization fails on transformers for one specific
reason: a few activation channels have magnitudes up to 100x larger than the
median channel, persistently across tokens, in specific layers (the effect
grows with model size and is strongest in the inputs to the MLP and attention
projections). A per-tensor activation scale set by the outliers leaves the
other channels with only a few effective quantization levels; per-token
scales help but the outlier channels still dominate within each token.

## The SmoothQuant migration

SmoothQuant observes that the difficulty can be moved. For a linear layer
Y = X W, insert a diagonal scaling: Y = (X diag(s)^{-1}) (diag(s) W), which
is mathematically identical. Choosing s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
shrinks the outlier activation channels (dividing X by s) and grows the
corresponding weight rows (multiplying W by s), which is safe because weights
have no outliers and quantize well anyway. The migration strength alpha is
the one knob: alpha = 0 leaves activations untouched, alpha = 1 pushes all
difficulty into the weights; alpha = 0.5 is the published default and the
usual optimum. After smoothing, both X and W quantize cleanly to INT8 and
W8A8 matches fp16 within noise on the paper's models up to OPT-175B.

## What smoothing costs and where it breaks

The scales s are computed once from a calibration set (a few hundred samples)
and folded into the preceding layer norm or linear layer, so inference cost
is zero. The breaks are predictable: models whose weight distributions also
have outliers (some gated MLP variants) cannot absorb the migrated
difficulty; alpha tuned on one domain transfers imperfectly to another; and
per-token dynamic activation quantization still outperforms static smoothed
quantization slightly, at kernel cost. SmoothQuant also does nothing for the
KV cache or for attention logits, which need their own treatment.

## FP8 training recipe

FP8 training at scale (H100 and later) keeps master weights in fp32 or bf16
and casts weights and activations to FP8 (E4M3) for the forward matmuls and
gradients to E5M2 or E4M3 for the backward, depending on the framework. The
critical machinery is scaling: each tensor gets a scale factor so its values
sit in the FP8 range, and because a fresh max computation per step is
expensive, delayed scaling reuses an amax history window (the maximum
absolute values observed over recent steps; the history length is a framework
setting, with values from tens to about a thousand steps in use). If a
tensor's distribution shifts suddenly, the stale scale clips values and the
loss shows it immediately.

## FP8 numerics in practice

Published large-scale FP8 runs (for example DeepSeek-V3's FP8 mixed precision
with fine-grained block scaling) report relative loss error under 0.25
percent versus bf16 baselines. The fine-grained part matters: DeepSeek-V3
used tile-wise 1x128 scaling for activations and 128x128 block scaling for
weights instead of one scale per tensor, which is what made E4M3 usable for
gradients as well. The failure mode to watch is not average error but
specific tensors: embedding gradients and the first and last layers are
commonly kept in bf16 even in aggressive FP8 recipes, and the LM head output
is computed in higher precision because small logit errors move the loss
directly.

## Choosing between INT8 smoothing and FP8

On H100-class hardware FP8 W8A8 is the default answer for new serving and
training because the tensor cores run it at 2x bf16 rate and no smoothing
calibration is needed. On older INT8-only hardware (T4, A10, most edge
accelerators), SmoothQuant-style W8A8 INT8 is the equivalent path. The
measurement that decides is end-to-end: tokens per second at the target
batch and sequence length plus a task suite delta against fp16, because a
method can win the kernel benchmark and lose the deployment on dequantization
or scaling overheads.

## Per-token dynamic quantization as the alternative

Instead of smoothing, activations can be quantized per token with a fresh
scale computed for each token at runtime. This adapts to whatever the
activation distribution does, including outlier tokens, at the cost of a
reduction per token per layer inside the kernel and less regular memory
access. Modern INT8 and FP8 kernels increasingly use per-token (or
per-token-group) dynamic scales because the hardware got fast enough at the
reduction; SmoothQuant's static smoothing remains the cheaper option on
older parts and composes with per-token scaling when both are available.
The empirical gap between static-smoothed and dynamic per-token at INT8 is
small on most published models, which is why the choice is usually made by
kernel availability rather than by quality.

## FP8 for inference serving

FP8 serving (W8A8 with per-tensor or block scales) on H100 roughly doubles
decode throughput over bf16 at bandwidth-bound batch sizes because the
weights halve in size, and improves prefill by up to the tensor-core ratio
when compute bound. The operational difference from training FP8: scales can
be calibrated once on representative traffic instead of tracked online, and
sensitive layers (LM head, first block) are routinely left in bf16. Engines
expose this as a quantization config; the measured quality delta on chat
suites is usually within noise at 8 bits, which is why FP8 became the
default serving precision on Hopper for large models while INT8 remains the
default on Ampere.

## Debugging FP8 training problems

When an FP8 run degrades, the suspects in order: stale scales after a data
distribution change (loss rises gradually, amax history shows clipping);
gradients cast to E4M3 where the recipe assumed E5M2 (noise floor rises,
small-gradient features stop learning); a tensor that should have stayed in
bf16 got swept into the FP8 config by a wildcard rule (often the embedding
or the final norm); and scale underflow on near-zero tensors such as
sparsely used expert weights in MoE models. All four are visible in the
amax and scale logs before they are visible in the loss, which is the
argument for logging quantization state as a first-class training metric
rather than treating FP8 as a transparent dtype swap.

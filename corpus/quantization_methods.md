# Quantization methods: PTQ, QAT, and number formats

## The affine quantization map

Integer quantization maps a real value x to an integer q with a scale s and
zero point z: q = round(x / s) + z, and dequantization is x_hat = s * (q - z).
For symmetric quantization z = 0 and s = max(|x|) / 127 for INT8. The scale is
where all the design freedom lives: one scale per tensor is cheapest and
worst; one per output channel is the standard for weights; one per group of
128 weights along the input dimension is the standard for 4-bit weight
compression, costing one fp16 scale per 128 weights (0.125 bits per weight of
overhead, plus the zero point if asymmetric). Finer groups track local ranges
better, so perplexity improves monotonically as group size shrinks from
whole-row to 128 to 64, with diminishing returns below 64.

## PTQ versus QAT

Post-training quantization (PTQ) quantizes a trained model using a small
calibration set (typically 128 to 512 samples) and no gradient updates to the
weights themselves, though methods like GPTQ do adjust the remaining
full-precision weights to compensate for quantization error. Quantization-
aware training (QAT) inserts fake-quantization into training so the weights
adapt to the grid; it gives the best low-bit quality but costs a training run.
At INT8, PTQ is effectively free of quality loss for most models and QAT is
rarely justified. At INT4 weight-only, PTQ with a good method is close. At
INT4 weights and activations, or INT2, QAT or distillation-based methods pull
clearly ahead, which is why sub-4-bit research models are usually trained,
not just calibrated.

## Weight-only versus weight-activation quantization

Weight-only quantization (W4A16, W8A16) dequantizes weights to fp16 in the
kernel and does the matmul in fp16. It wins when the workload is memory bound,
which decode-time inference is: a 7B model at INT4 is about 3.5 GB of weights
instead of 14 GB, so tokens per second at batch 1 roughly tracks the 4x
smaller read. Weight-activation quantization (W8A8, W4A8) also speeds up the
matmul itself, which matters for prefill and large-batch serving where the
work is compute bound, but activations are much harder to quantize than
weights because of outlier channels, which is the entire subject of
SmoothQuant-style methods.

## INT8 with outlier handling

LLM.int8() showed that transformer activations develop a small number of
outlier feature dimensions (about 0.1 percent of dimensions) whose magnitudes
are up to 20x larger than the rest, emerging reliably around 6.7B parameters.
A single per-tensor activation scale is destroyed by these outliers. The fix
is a mixed decomposition: the outlier dimensions (identified by a threshold,
6.0 in the paper) are computed in fp16 and the remaining 99.9 percent in
INT8, then summed. Quality matches fp16 closely. The systems cost is two
matmuls and a gather, which is why later kernels prefer per-channel smoothing
or FP8 over decomposition on newer hardware.

## FP8 formats: E4M3 and E5M2

FP8 comes in two standard variants. E4M3 has 4 exponent bits and 3 mantissa
bits: range up to about 448, relative precision about 6 percent, used for
weights and activations in the forward pass. E5M2 has 5 exponent bits and 2
mantissa bits: range up to about 57344 like fp16, precision about 12 percent,
used for gradients where range matters more. Neither has the range to be used
without scaling: production FP8 training keeps per-tensor (or per-block)
scales, often with delayed scaling that reuses the maximum absolute value
history (amax) from previous steps because computing a fresh scale every step
would stall the pipeline. H100 and later tensor cores execute FP8 at 2x the
bf16 FLOPS, which is the entire motivation.

## Calibration and error measurement

PTQ quality is decided by the calibration set more often than by the format.
Calibration data should look like deployment data in domain and sequence
length; calibrating a chat model on raw web text measurably worsens chat
perplexity after quantization. The honest error metrics are task-level:
perplexity on held-out text plus a downstream suite, reported against the
fp16 baseline of the same model. Rules of thumb from published results: INT8
weight-only is within 0.1 perplexity almost always; INT4 group-128 with GPTQ
or AWQ is within about 0.2 to 0.5 on 7B models and less on larger ones; the
gap grows sharply below 4 bits and on small models, which have less
redundancy to absorb the error.

## Mixed-precision layer assignment

Not all layers quantize equally. Sensitivity analyses consistently find the
first and last transformer blocks, the embedding, and the LM head more
sensitive than middle blocks, and within blocks the down-projection of the
MLP is a frequent outlier. A common production recipe keeps embeddings and
the LM head at fp16, the first and last block at INT8, and the middle at
INT4, buying back most of the INT4 quality loss for a few percent of the
memory saving. Sensitivity is measured per layer by quantizing one layer at
a time and recording the perplexity delta, a one-afternoon experiment that
should precede any serious low-bit deployment, because the sensitive layers
are model-specific enough that copying another model's assignment is a
guess.

## Quantization at long context

Weight quantization interacts with context length through the KV cache and
through activation ranges: activation outliers grow with sequence length in
several model families, so a W8A8 calibration done at 2048 tokens can
under-cover the ranges seen at 32768. Calibrate at the deployment sequence
length, or use per-token dynamic activation scales that adapt by
construction. Separately, at long context the KV cache dominates memory, so
weight-only INT4 stops being the binding optimization and cache quantization
takes over; a serving plan that quantizes weights and ignores a 100 GB cache
has optimized the smaller term.

## Reporting quantization results

A quantization claim needs five numbers to be checkable: the format and
method (INT4 GPTQ group 128, not "4-bit"), the calibration set, the baseline
(fp16 of the same checkpoint), the quality delta on a named suite, and the
speed or memory measurement on named hardware at a stated batch size and
sequence length. Missing any one of these, the claim cannot be reproduced
and should be treated as marketing. This standard is cheap to meet and is
the difference between the quantization literature, which mostly meets it,
and quantization product pages, which mostly do not.

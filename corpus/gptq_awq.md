# GPTQ and AWQ: how modern PTQ methods work

## GPTQ: error compensation with the Hessian

GPTQ quantizes one layer at a time, one column at a time, and compensates for
each column's quantization error by adjusting the still-unquantized columns.
The compensation uses the inverse Hessian of the layer's least-squares
objective, H = 2 * X X^T computed from calibration activations X: when column
q is rounded to the grid, the error is pushed onto the remaining columns in
proportion to the corresponding entries of H^{-1}. Solving for all columns
with a fresh inverse per column would be far too slow, so GPTQ uses the
Optimal Brain Quantization reformulation: quantize in a fixed order, use the
Cholesky factorization of H^{-1} to get every needed term, and apply updates
lazily in blocks of 128 columns. That makes a 175B model quantizable in about
4 GPU hours in the original paper.

## GPTQ practical settings and failure modes

The standard recipe is group size 128, calibration on 128 sequences of 2048
tokens from a general corpus, and activation-order (act-order) permutation:
quantizing columns in decreasing order of diagonal Hessian magnitude (columns
whose inputs are large get quantized first) reduces perplexity by a small but
consistent margin at 4 bits and a larger one at 3 bits. Failure modes are
concrete: error accumulates toward the last layers, so a few layers (often
the final MLP blocks) are much more damaged than average; calibration on the
wrong domain shifts which columns look important; and at 2 bits the method
collapses entirely (perplexity in the hundreds) regardless of settings. GPTQ
is weight-only: it says nothing about activation quantization.

## AWQ: protect the salient weights

AWQ starts from a different observation: weight importance is determined by
the activations, not the weights. About 1 percent of weight channels are
salient, identified by the average magnitude of the corresponding input
activations on calibration data. Instead of keeping those weights in higher
precision, AWQ scales them up by a factor s (and scales the corresponding
activations down by the same factor, which is exact in a linear layer):
Y = (X diag(s)^{-1}) (diag(s) W). Scaling a salient channel by s > 1 reduces
its relative quantization error because the group scale is set by the largest
element; the optimal s per channel group is found by a small grid search
minimizing the layer output error on calibration data. No backpropagation and
no error compensation loop, which makes AWQ faster to run than GPTQ.

## AWQ results and kernel story

With group size 128 and 4 bits, AWQ matches or slightly beats GPTQ on
perplexity in most published comparisons and is notably better on some
instruction-tuned and multimodal models, because the salience estimate is
data-driven rather than weight-driven. AWQ ships with fused kernels (the
GEMM and GEMV paths in the AWQ repo, and Marlin-based kernels in vLLM) that
dequantize on the fly; measured decode speedups on a 7B model are about 2x to
3x over fp16 at small batch, consistent with the 4x smaller weight read minus
dequantization overhead. The method's limit is the same as GPTQ's: it is a
weight-only PTQ method, so it cannot fix activation outliers, and quality at
3 bits and below still degrades visibly.

## GPTQ versus AWQ, decided by measurement

On a typical 7B to 13B model at 4 bits group 128, the two methods land within
about 0.1 perplexity of each other, and the choice is usually made on tooling:
AWQ has the faster maintained kernels in vLLM; GPTQ has wider support in
older tooling and an act-order variant that wins on some base models. Both
beat naive round-to-nearest INT4 by a wide margin (often 1 to 3 perplexity
points at 7B), which is the comparison that matters: the format is not the
method. Anyone reporting "INT4 quantization" without naming the method and
group size has not given enough information to reproduce the number.

## Calibration set sensitivity

Both methods are only as good as the activations the calibration set
produces. GPTQ's Hessian and AWQ's salience both come from the same kind of
pass: 128 sequences is the common budget, and results move measurably when
calibration data is off-domain (calibrating a code model on news text costs
visible perplexity on code). Sequence length matters too: calibrating at
512 tokens and deploying at 8192 underrepresents the activation ranges of
long contexts. The sound practice is calibration data sampled from the
deployment distribution at the deployment length, refreshed when the model
or the traffic changes. This is also why quantized checkpoints distributed
without calibration details are hard to trust: the same method name hides a
variable that moves the result.

## Kernel implementations

A GPTQ or AWQ checkpoint is only fast if the serving kernel dequantizes
efficiently. The Marlin kernel (used by vLLM for both formats) repacks
weights offline into a layout where the dequantization and the tensor-core
matmul pipeline overlap, reaching close to the theoretical 4x memory-bound
speedup at small batch. Older kernels that dequantize into a temporary fp16
buffer lose much of the advantage and can be slower than fp16 at large
batch, where dequantization is pure overhead on a compute-bound problem.
Benchmark the exact kernel and version you will deploy; "AWQ model" names a
checkpoint format, not a speed.

## Composing the two ideas

GPTQ's error compensation and AWQ's activation-aware scaling attack the same
error from different sides, and they compose: scale the weights by salience
first, then run GPTQ's compensated rounding on the scaled matrix (the
combination appears in several later papers and in some production recipes).
The gains over either alone are small at 4 bits (hundredths to tenths of a
perplexity point) and larger at 3 bits, where every mechanism matters. Below
3 bits both families give way to trained methods (vector quantization with
fine-tuning, or QAT), and the honest report at those widths is a quality
curve, not a single number.

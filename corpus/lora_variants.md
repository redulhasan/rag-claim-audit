# LoRA variants: DoRA, AdaLoRA, rsLoRA, LoRA+, and VeRA

## DoRA: magnitude and direction

DoRA decomposes a pretrained weight column-wise into a magnitude scalar and a
direction matrix: W = m * V / ||V||, where m is a trainable vector with one
entry per output column and V is adapted with a standard LoRA update. The
motivation is an analysis showing full fine-tuning changes magnitude and
direction in a correlated way that plain LoRA (which only learns an additive
low-rank direction change) imitates poorly. Training only m (d parameters per
matrix) plus the LoRA pair recovers most of the gap: the DoRA paper reports
matching or beating LoRA at the same trainable parameter count on commonsense
reasoning and matching full fine-tuning more closely on several tasks. At
inference the adapted weight folds back into a single matrix, so DoRA has the
same zero added latency as LoRA.

## AdaLoRA: adaptive rank budgets

AdaLoRA parametrizes the update in SVD form, delta_W = P * Lambda * Q, with
orthogonality regularization on P and Q, and prunes singular values during
training according to an importance score that combines the magnitude of each
singular value with the sensitivity of the loss to it. Training starts with a
higher total rank budget and anneals to the target budget (for example from
1.5x the final budget), so rank is reallocated from matrices that do not need
it (often the MLP down-projections) to matrices that do (often attention q and
v in lower layers). The gain over uniform-rank LoRA is largest at very small
budgets (r = 2 to 4 average), where a uniform allocation wastes parameters;
at r = 16 and above the difference mostly disappears.

## rsLoRA: fixing the scale

Standard LoRA scales the update by alpha / r. Rank-stabilized LoRA argues this
is the wrong exponent: with alpha / r, the output variance of the update
shrinks as r grows, so higher ranks learn less per step and large-r LoRA
underperforms. rsLoRA uses alpha / sqrt(r) instead, which keeps the update
scale stable across ranks. The practical consequence: with rsLoRA, raising r
from 16 to 64 actually helps on harder tasks, whereas with the classic scaling
it often does nothing or hurts unless alpha is retuned. The change is one
line, and several training frameworks expose it as a flag (use_rslora).

## LoRA+: different learning rates for A and B

LoRA initializes A with random Gaussian entries and B to zero, so at step 0
the gradients flowing into A and B have very different scales. LoRA+ shows
that in the wide-network limit the efficient choice is a much larger learning
rate for B than for A, with a ratio lambda = lr_B / lr_A of about 2^4 = 16 in
their experiments. Using the same learning rate for both (the default in most
frameworks) leaves performance on the table, especially at larger ranks and
widths; LoRA+ reports the same loss with 1.5x to 2x fewer steps on several
benchmarks at no extra memory cost. The fix is a per-matrix learning rate in
the optimizer param groups.

## PiSSA and structured initialization

PiSSA initializes A and B from the principal singular vectors and values of
the pretrained weight itself (delta starts as the top-r SVD component of W,
and the residual W - delta becomes the frozen base) instead of random A and
zero B. Because the adapter starts aligned with the directions the weight
already uses most, convergence is faster in the first epochs and final loss is
slightly lower than random init in the paper's experiments. OLoRA uses a QR
decomposition for a similar effect. Initialization matters most in the
low-step regime (a few hundred steps); with long training the gap narrows.

## VeRA: shared random matrices

VeRA pushes parameter efficiency further: a single pair of random matrices B
and D is shared across all layers and frozen, and each layer learns only two
scaling vectors that multiply the rows and columns. Trainable parameters drop
by about 10x versus LoRA at the same rank (VeRA uses ranks like 256 with only
vectors trained), and the shared matrices can be regenerated from a seed, so
the stored adapter is just the vectors. The cost is capacity: VeRA needs the
higher rank to match LoRA quality, and per-layer adaptation is limited to
rescaling a shared subspace, which underperforms on tasks needing very
different updates per layer.

## Choosing among variants

At r >= 16 with a normal training budget, plain LoRA with rsLoRA scaling and a
LoRA+ learning-rate ratio captures most of the available gain for free. DoRA
is worth it when matching full fine-tuning closely matters and the small extra
training compute (the normalization per step) is acceptable. AdaLoRA earns its
complexity only at tiny rank budgets. VeRA is for storing or serving thousands
of adapters where adapter size, not quality, is the constraint.

## Merging, serving, and multi-adapter effects

All LoRA-family adapters merge into the base weight for single-adapter
serving (W + (alpha/r) B A), so inference cost is identical across variants
except DoRA, whose magnitude vector also folds in exactly. Multi-adapter
serving (S-LoRA, Punica-style batched adapter kernels) changes the ranking:
adapter size and the ability to batch many different adapters in one forward
pass dominate, which favors small uniform adapters and penalizes variants
with per-layer rank variation (AdaLoRA) whose kernels are harder to batch.
VeRA's tiny adapters are the extreme point of this trade. If the deployment
serves one adapter, choose by quality; if it serves thousands, choose by
adapter bytes and kernel support, and expect the quality ranking to shuffle.

## What the rank allocation evidence says

Across AdaLoRA-style importance measurements and independent ablations, the
consistent findings are: attention projections in lower and middle layers
use rank more effectively than MLP up-projections; the final layers often
need less rank than intuition suggests; and v projections benefit slightly
more than q at equal rank. These are tendencies measured on a handful of
models, not laws, and they invert on some task families (code models shift
demand toward MLP layers). Treat them as initialization for a budget search
on your own task, measured by validation loss per trainable parameter, not
as a recipe.

## Initialization and scaling pitfalls

The recurring silent bugs in LoRA training are scaling bugs: alpha left at a
default while r changes (the effective step size scales with alpha / r, so
doubling r at fixed alpha halves the update scale), dropout applied to the
adapter path in training but the merged weight used at inference without
rescaling expectations, and adapters trained on one base revision merged
into another. rsLoRA exists precisely because the first bug is so common.
The check is cheap: log the Frobenius norm of delta_W relative to W per
layer after training; healthy adapters land in a band around 1 to 10 percent
of the base norm, and outliers in that table are almost always configuration
errors rather than interesting model behavior.

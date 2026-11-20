# LoRA and parameter-efficient fine-tuning

## Low-rank adaptation

LoRA freezes the pretrained weight W (d x k) and learns a low-rank update
delta_W = B A, where B is d x r, A is r x k, and the rank r is much smaller
than d and k. The forward pass becomes h = W x + (alpha / r) B A x.

Trainable parameter count drops from d*k to r*(d + k). For a 4096 x 4096
matrix with r = 16, that is 16.7M parameters down to 131k, about 0.8 percent
of the original matrix. In practice LoRA is applied to the attention
projections (q, v most often, sometimes all linear layers) and occasionally
to the MLP blocks.

## Rank r vs alpha

r controls the capacity of the update: how many directions the weight change
can use. alpha is a plain scaling constant, applied as alpha / r, so raising
alpha with r fixed just scales the update up. A common starting point is
r = 16 and alpha = 32, giving a scale of 2. Doubling r and alpha together
keeps the initial scale the same but adds capacity. If training is unstable,
lowering alpha (or the learning rate) is usually the first knob, not r.

There is no rank that is best in general. r = 4 to 8 is often enough for a
narrow style or format task; tasks that need new factual behaviour or a new
language usually need r = 32 or higher, and at some point full fine-tuning
is simply the honest option.

## QLoRA and NF4

QLoRA stores the frozen base weights in 4-bit NormalFloat (NF4) and trains
LoRA adapters in bf16 on top. NF4 is a quantile quantization scheme: its
levels are placed at quantiles of a normal distribution, which matches the
roughly normal distribution of pretrained weights better than uniform 4-bit
bins. A 65B model at 4 bits needs about 65e9 * 0.5 bytes = 32.5 GB for the
weights, versus 130 GB in fp16, which is what makes single-GPU fine-tuning
of that size possible at all.

QLoRA also uses double quantization (quantizing the quantization constants
themselves) and paged optimizers to handle gradient checkpointing memory
spikes. The adapters, not the base, carry all of the task-specific change,
so merging a QLoRA adapter back into a dequantized base is approximate.

## Memory math for training

Rough budget for full fine-tuning with AdamW in mixed precision: weights
2 bytes/param (fp16 copy) + gradients 2 bytes/param + Adam moments
8 bytes/param (fp32 m and v) + fp32 master weights 4 bytes/param, so around
16 bytes per parameter before activations. A 7B model needs roughly 112 GB,
which is why full fine-tuning of 7B does not fit on a single 80 GB card
without optimizer sharding or 8-bit optimizers.

With LoRA the optimizer states only cover the adapter parameters, so the
same 7B model trains in the 20 to 30 GB range depending on sequence length
and batch size. The frozen weights still have to sit in memory; LoRA saves
optimizer and gradient memory, not the forward pass.

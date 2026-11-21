# QLoRA in depth: NF4, double quantization, and paged optimizers

## NF4: a quantile-based 4-bit format

NF4 (4-bit NormalFloat) is built on the observation that pretrained weights
are approximately normally distributed per block. A fixed 4-bit float format
wastes codes on the tails; NF4 instead places its 16 code values at the
quantiles of a standard normal, so each code is used about equally often for
normally distributed data. The values are asymmetric around zero (more codes
on the negative side in the published table) and include an exact zero, which
matters because padding and pruned weights are exactly zero. Quantization is
blockwise: each block of 64 weights shares one fp32 scale constant (the
absolute maximum of the block), and dequantization is a table lookup times the
scale.

## Double quantization arithmetic

The block scales themselves add up: with block size 64 and fp32 constants,
the overhead is 32 / 64 = 0.5 bits per parameter, so NF4 alone costs 4.5 bits
per parameter in total. Double quantization quantizes the scale constants
themselves: the fp32 scales are grouped into blocks of 256 and stored in
8-bit floats with a second-level fp32 constant. The overhead becomes
8 / 64 + 32 / (64 * 256) = 0.125 + 0.002, about 0.127 bits per parameter, for
a total of about 4.13 bits per parameter. On a 65B model that saves roughly
(0.5 - 0.127) * 65e9 / 8 bytes, about 3 GB, compared to single-level NF4,
which is the difference between fitting and not fitting on a 48 GB card once
optimizer state and activations are counted.

## Memory budget for a 65B model

The headline QLoRA result is fine-tuning a 65B model on one 48 GB GPU. The
budget: 65e9 parameters at about 4.13 bits is about 33.6 GB of base weights.
LoRA adapters on all linear layers at r = 64 are a few hundred MB in bf16.
Gradients are only needed for the adapters. The 8-bit Adam states for the
adapter parameters are small. What remains is activations, which are
controlled with gradient checkpointing and a modest sequence length and batch
size. The base weights stay in NF4 during the forward pass; computation
dequantizes block by block to bf16 on the fly, so the memory peak includes one
dequantized block buffer, not a full bf16 copy.

## Paged optimizers

Even with the budget above, activation spikes at long sequence lengths can
push memory over the edge mid-training and kill a run hours in. Paged
optimizers use CUDA unified memory to page optimizer state out to CPU RAM
when a spike hits, the same paging idea as PagedAttention applied to training.
In the QLoRA setup this specifically guards the moments when a long example
enters the batch. It costs nothing in the normal case and converts an
out-of-memory crash into a slow step.

## LoRA configuration used by QLoRA

Two configuration choices in the paper matter more than they look. First,
LoRA is applied to all linear layers, not only q and v: restricting to
attention projections measurably hurt quality at fixed adapter size. Second,
rank mattered less than expected once all layers were adapted; r = 64 with
alpha = 16 (a scaling of 0.25) was the paper's main configuration, and r = 16
was close behind on several tasks. The learning rate was 1e-4 to 2e-4 for the
larger models with a constant schedule, notably higher than typical full
fine-tuning rates, because only the adapters move.

## Quality evidence and limits

The paper's check is that 4-bit QLoRA matches 16-bit LoRA and full
fine-tuning on the same tasks (MMLU within noise for 7B to 65B models tuned
on the same data), and that NF4 beats plain INT4 (FP4) at the same bit width,
with double quantization recovering the small remaining gap. The Guanaco
models trained this way reached about 99 percent of ChatGPT's score on the
Vicuna benchmark as judged by GPT-4, a number that says as much about the
judge as the model. The real limits are speed (dequantization makes QLoRA
training slower per step than bf16 LoRA, roughly 1.3x to 1.5x in reported
setups) and the frozen base: tasks needing the base weights themselves to
move, like learning a new language's script, are outside what any adapter on a
frozen quantized base can do.

## Dequantization cost in the forward pass

NF4 weights are stored packed (two 4-bit values per byte) and must be
dequantized to bf16 before the matmul. The dequantization is a gather from
the 16-entry NF4 table times the block scale, plus the second-level
dequantization of the scales themselves under double quantization. This runs
on every forward pass over every weight, so QLoRA training steps are slower
than bf16 LoRA steps at the same shape; reported slowdowns cluster around
1.3x to 1.5x, growing at small batch sizes where the fixed dequantization
work amortizes worst. Inference from an NF4 checkpoint without merging has
the same cost, which is why deployment usually merges the adapter and
re-quantizes with a serving-oriented method (GPTQ or AWQ) rather than
serving the training format directly.

## NF4 versus plain INT4, measured

The ablation in the QLoRA paper isolates the format: at 4 bits on the same
models and tasks, NF4 beats FP4 (a plain 4-bit float grid) and INT4
(uniform) on perplexity and MMLU, and NF4 with double quantization is within
noise of bf16 LoRA at 7B to 33B and slightly ahead of FP4 at 65B in their
runs. The margin is small in absolute terms (tenths of a perplexity point)
but consistent, and it comes from the quantile placement: a uniform grid
spends half its codes on the tails of the weight distribution where almost
no weights live. The lesson generalizes: at very low bit widths the code
placement matters as much as the bit count, which is also the premise of
later learned-codebook methods.

## What QLoRA cannot fix

The base model is frozen and quantized, so three limits are structural.
Knowledge the base model does not have cannot be added beyond what a
low-rank update on each layer can express; domain adaptation requiring
broad weight movement (a new language with a new script is the standard
example) underperforms full fine-tuning. Quantization error is baked in: if
the 4-bit base already lost a capability, the adapter trains around the loss
but rarely restores it. And the memory math assumes adapters on all layers
with modest rank; pushing rank to several hundred to chase capacity erodes
the parameter-efficiency that justified the method, at which point bf16 LoRA
with ZeRO offload is usually the better tool.

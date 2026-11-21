# Activation checkpointing

## The basic trade

Activation checkpointing (gradient checkpointing) stores only the inputs of
selected segments of the network during the forward pass and recomputes the
segment's internal activations during the backward pass. For a network split
into sqrt(N) segments of sqrt(N) layers each, stored activations drop from
O(N) layers' worth to O(sqrt(N)): the segment boundaries plus one segment
being recomputed at a time. The cost is one extra forward pass over the
recomputed segments, which is about 33 percent more compute in the
forward-plus-backward accounting (a normal step costs roughly 1 forward + 2
forward-equivalents of backward; checkpointing makes it 2 + 2).

## Full versus selective checkpointing

Full checkpointing wraps every transformer block. Selective checkpointing
recomputes only the memory-heavy, compute-cheap parts: in Megatron's
selective mode, the attention internals (the softmax statistics, the
attention matrix products' inputs, dropout masks) are recomputed while the
expensive matmul outputs are stored. Attention internals scale with
sequence^2 * heads in naive form and with sequence * hidden in fused form,
so at long context they dominate activation memory while costing little to
recompute. Selective checkpointing typically recovers most of the memory
saving of full checkpointing (often 70 to 90 percent of it at long sequence
lengths) at a few percent of step time instead of 33 percent.

## Memory numbers at long context

Take batch 1, sequence 8192, hidden 4096, 32 layers, bf16 with fused
attention. Storing everything costs on the order of 34 bytes per token per
hidden unit per layer across all the saved tensors, roughly 37 GB total.
Full checkpointing at block granularity stores 32 block inputs
(8192 * 4096 * 2 bytes = 67 MB each, about 2.1 GB) plus one block's internals
at a time during backward. That is the difference between fitting and not
fitting next to a sharded 7B model's state on an 80 GB card. The saving
grows linearly with sequence length while the recompute cost grows with the
attention FLOPs, so the trade gets better, not worse, as context grows.

## Interaction with FSDP and ZeRO-3

Checkpointing composes with sharding but the order of operations matters:
with FSDP or ZeRO-3, a checkpointed segment's parameters must be
all-gathered again for the recomputation during backward, adding
communication that the non-checkpointed backward would not need. Frameworks
handle this by keeping the all-gathered parameters alive from forward to
backward for non-checkpointed units and re-gathering for checkpointed ones.
The practical consequence: enabling full checkpointing on top of ZeRO-3 costs
more than the nominal 33 percent on bandwidth-limited setups, and selective
checkpointing is the usual compromise there.

## Offloading activations

The alternative to recompute is to move activations to CPU memory over PCIe
during the forward pass and bring them back for the backward. At PCIe 4.0
x16 speeds (about 32 GB/s) offloading 37 GB costs over a second each way,
which only hides under compute if the step is long; on NVLink-connected CPU
memory (Grace-Hopper) the same transfer is about 15x faster and offloading
becomes genuinely competitive with recompute. Some stacks do both: offload
the segment boundaries, recompute the internals. The decision is a bandwidth
calculation against the machine's actual links, not a preference.

## When not to checkpoint

If activations fit comfortably (short sequences, small batch, large memory),
checkpointing is a pure 33 percent tax for nothing; profile first. It also
interacts badly with some pipeline schedules by changing the forward time
profile the schedule was tuned for, and with dropout it requires the RNG
state per segment to be saved and restored exactly, which older frameworks
occasionally got wrong in ways that silently changed results. The standard
practice at scale: selective checkpointing on by default for context lengths
above 8k, full checkpointing when memory still does not fit, off otherwise.

## Tuning checkpoint granularity

The sqrt(N) rule is the theory optimum for uniform layers; practice uses
one transformer block as the segment because it aligns with module
boundaries and FSDP wrap units. Finer granularity (checkpointing inside the
block, around attention and MLP separately) saves a bit more memory at more
recompute launches; coarser (two blocks per segment) saves less with fewer
launches. The measurable knob is peak memory versus step time on your
shapes: try block-level first, and only go finer if the remaining peak is
inside one block's internals, which happens at very long sequence lengths
where a single block's attention activations exceed the rest of the budget.

## Profiling before enabling

The decision to checkpoint should come from a memory profile, not a guess:
capture the peak memory breakdown (parameters, optimizer, gradients,
activations) for one step at production shapes. If activations are under
about a third of the total, checkpointing buys little and costs its 33
percent (or selective's few percent) for nothing; the money is better spent
on batch size. If activations dominate, checkpointing is the cheapest memory
in the system. Profilers (PyTorch's memory timeline is adequate) make this a
ten-minute check, and it also catches the embarrassing case where the peak
is not activations at all but a logits tensor at large vocabulary and batch,
which checkpointing does nothing for.

## Interaction with pipeline schedules

In pipeline parallelism, checkpointing changes each stage's forward time,
and schedules tuned for the uncheckpointed profile can develop larger
effective bubbles because the recompute happens inside the backward window
the schedule reserved. Interleaved schedules are more sensitive to this than
1F1B. The practical fix is to re-measure the bubble after enabling
checkpointing rather than assuming the nominal (p-1)/(m+p-1), and to raise
the microbatch count m if the bubble grew, memory permitting, which it
usually does precisely because checkpointing freed memory.

# ZeRO optimizer stages 1 to 3

## The 16 bytes per parameter accounting

Mixed-precision Adam training stores, per parameter: 2 bytes of bf16 weights
used in the forward pass, 2 bytes of bf16 gradients, and 12 bytes of fp32
optimizer state (fp32 master weights 4, Adam first moment 4, Adam second
moment 4). Total 16 bytes per parameter before activations. A 7B model needs
about 112 GB, a 70B model about 1.12 TB. ZeRO partitions this state across
data-parallel ranks so per-device memory falls with the number of devices;
the three stages differ in how much of the 16 bytes they partition.

## ZeRO stage 1: partition the optimizer state

Stage 1 partitions only the 12 bytes of optimizer state (master weights and
Adam moments). Each rank owns 1/N of the fp32 state, updates its partition,
and the refreshed bf16 weights are gathered for the next step. Per-device
memory drops from 16 to about 4 + 12/N bytes per parameter (the 2 bytes of
weights and 2 bytes of gradients stay replicated). A 7B model on 8 devices
needs about 5.5 bytes per parameter, roughly 38.5 GB per device, which fits
an 80 GB card with room for activations. A 70B model at the same setting
needs about 385 GB per device, so stage 1 alone cannot train it; that is
what stages 2 and 3 are for. Communication volume is unchanged from plain
data parallel (about 2P bytes for the gradient synchronization).

## ZeRO stage 2: partition gradients too

Stage 2 also partitions gradients: each rank keeps only the gradient shard
for the parameters it owns (a reduce-scatter instead of an all-reduce), so
the 2 bytes of gradients divide by N as well. Per-device memory becomes about
2 + 14/N bytes per parameter. Communication volume is the same as stage 1
(reduce-scatter moves the same bytes as all-reduce for this pattern).
Stage 2 is the default for many DeepSpeed recipes because it removes most
memory pressure for models up to roughly 20B on typical node counts without
paying stage 3's extra parameter communication.

## ZeRO stage 3: partition the parameters

Stage 3 partitions the bf16 weights as well: per-device memory approaches
16/N bytes per parameter plus the transient full layer being computed. A 70B
model on 64 devices needs about 16 * 70e9 / 64 = 17.5 GB per device of state.
The cost is communication: parameters must be all-gathered before each
layer's forward and backward, adding about 1.5x the parameter volume in
traffic per step compared to data-parallel gradient sync (3P versus 2P in the
usual accounting). Prefetching the next layer's parameters during the current
layer's compute hides much of this on fast interconnects. Stage 3 also
enables the model-parallel-free training of models whose weights alone exceed
device memory, which stages 1 and 2 cannot do.

## ZeRO-Offload and ZeRO-Infinity

ZeRO-Offload moves the partitioned optimizer state (and optionally gradients)
to CPU memory and runs the optimizer step on the CPU; ZeRO-Infinity extends
the same idea to NVMe storage. These trade bandwidth on the PCIe or NVMe
link for GPU memory, and they make sense when the alternative is not training
at all: single-GPU fine-tuning of 10B+ models is possible with offload at
roughly 2x to 4x slower step time. The optimizer step on CPU is not the
bottleneck people expect (it is elementwise); the bottleneck is moving
gradients down and parameters up each step.

## Choosing a stage

The decision is arithmetic, not taste. Compute state per device for each
stage from the 16 bytes formula and the device count, add activations (with
checkpointing, roughly proportional to batch * sequence * hidden * layers
divided by the checkpointing factor), and pick the lowest stage that fits
with about 20 percent headroom. Going one stage higher than needed costs
real throughput: stage 3's extra all-gathers typically cost 10 to 30 percent
of step time versus stage 2 on the same hardware when both fit. Report the
stage in any training write-up; "trained with DeepSpeed" without the stage
is as incomplete as "quantized" without the bit width.

## ZeRO versus FSDP, honestly

ZeRO-3 and FSDP implement the same idea (shard everything, gather per layer)
and differ in ecosystem and details: DeepSpeed's ZeRO has the offload and
NVMe extensions and a longer history in very large runs; PyTorch FSDP is
native to the framework, composes better with torch.compile, and is the
default in most open training code today. Measured throughput differences on
identical hardware are usually within noise once both are tuned; the
decision is made on the surrounding stack (checkpointing tools, optimizer
support, debugging familiarity). Anyone claiming a large intrinsic speed
difference between them is usually comparing a tuned setup against a default
one.

## Checkpointing under partitioning

Sharded training complicates checkpoints: the full model state exists
nowhere in memory, so saving requires either gathering the state onto one
rank (slow, memory-spiky, but produces a standard checkpoint) or saving
per-rank shards (fast, and the checkpoint is tied to the same world size
and partitioning unless resharded on load). Frameworks support both; the
sharded format plus a resharding loader is the scalable choice, with a
periodic full checkpoint for portability. Optimizer state dominates
checkpoint size under Adam (12 of the 16 bytes per parameter), so training
checkpoints are roughly 6x the size of the bf16 inference weights, a number
that belongs in the storage budget for any long run.

## Elasticity and fault tolerance

At hundreds of devices, hardware failures during a multi-week run are
certain, and partitioned state changes the recovery story: losing one rank
loses 1/N of every tensor, so recovery means restarting from the last
checkpoint with all ranks, not patching one replica. Checkpoint frequency
is therefore set by failure statistics (mean time between failures versus
checkpoint write time), and asynchronous checkpointing, which snapshots to
CPU memory and writes in the background, exists to make frequent checkpoints
cheap. ZeRO and FSDP both sit under this same operational reality; the
partitioning math decides memory, the failure math decides the checkpoint
schedule.

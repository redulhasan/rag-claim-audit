# Data parallel training: DDP and FSDP

## DDP gradient synchronization

PyTorch DDP replicates the full model on every GPU and synchronizes gradients
with an all-reduce each step. Communication volume per step is about 2 *
(N - 1) / N * P bytes for P parameters in a ring all-reduce, approaching 2P
for large N. To hide this cost, DDP buckets gradients (default bucket size
25 MB) and starts reducing each bucket as soon as its gradients are ready
during the backward pass, overlapping communication with computation.
Replication is the constraint: weights, gradients, and optimizer states live
in full on every device, so DDP alone caps model size at single-GPU memory
(about 16 bytes per parameter with Adam and mixed precision: a 7B model needs
about 112 GB before activations, already over one 80 GB card).

## FSDP sharding

FSDP shards parameters, gradients, and optimizer states across devices. Each
device stores 1/N of the state; before a layer's forward or backward, its
full parameters are all-gathered, used, and released. Communication per step
is roughly 1.5x the parameter count in bytes moved per device (all-gather in
forward, all-gather again in backward under the default reshard-after-forward
policy, plus a reduce-scatter of gradients), versus DDP's 2x for gradients
only but with full state resident. The trade is explicit: FSDP spends more
bandwidth to divide memory by N. On fast interconnects (NVLink, InfiniBand)
this is the right trade for models above about 3B parameters on 80 GB cards;
on slow Ethernet the all-gathers can dominate and smaller-scale DDP wins.

## Sharding units and wrap policy

FSDP wraps the model in nested units, typically one transformer block per
unit. The wrap granularity sets the memory-versus-communication balance:
wrapping the whole model as one unit means all-gathering every parameter
before the forward pass (peak memory equals the full model, defeating the
point); wrapping too finely adds kernel launch and communication overhead per
tiny unit. The auto-wrap policy based on transformer block classes is the
standard choice. Flat parameters within a unit are stored as one contiguous
buffer, which makes the all-gather a single collective per unit.

## Mixed precision and state dtypes

FSDP's mixed precision policy has three dtypes to choose: parameter dtype for
computation (bf16), reduce dtype for gradient communication (bf16 or fp32),
and buffer dtype. The memory accounting that matters: with N devices, per-
device state is (params + grads + Adam states) / N plus the transient full
parameters of the unit currently computing. Keeping gradient reduction in
fp32 improves stability for large runs at some bandwidth cost; most open
recipes reduce in bf16 and accept it. Optimizer states stay in fp32 shards
in the standard configuration, so Adam's 8 bytes per parameter (two fp32
moments) divide by N like everything else.

## HSDP and topology awareness

Hybrid sharding (HSDP) shards within a node or rack and replicates across
them: shard group size is set to the number of GPUs sharing fast interconnect
(for example 8 per node), and gradients are all-reduced across replicas.
This bounds the all-gather traffic to the fast fabric and pays the cheaper
DDP-style synchronization on the slow fabric. For multi-node training of
models that fit sharded-per-node, HSDP is usually faster than full FSDP at
the same memory per device times the replica count. The configuration knob is
the shard group size, and the right value is a topology question, not a model
question.

## When DDP is still the right answer

If the model plus optimizer fits comfortably on one device (roughly up to
3B parameters on 80 GB with Adam, more with 8-bit optimizers or Adafactor),
DDP is simpler and faster: no per-layer collectives, no wrap policy to tune,
and gradient bucketing overlaps well. FSDP's complexity earns its keep only
when memory forces it. The common production pattern for mid-size models is
DDP with activation checkpointing and a memory-efficient optimizer, scaling
data parallel width for throughput rather than sharding for capacity.

## Gradient accumulation interplay

Gradient accumulation splits a global batch into micro-steps whose gradients
are averaged before the optimizer step. With DDP the trap is synchronization
on every micro-step: the no_sync context manager defers the all-reduce until
the last micro-step, cutting communication by the accumulation factor. With
FSDP the equivalent control exists but interacts with resharding: parameters
gathered for a micro-step's backward may be released and re-gathered per
micro-step, so accumulation under FSDP can cost more communication than the
naive model suggests. The effective batch arithmetic also matters for
reporting: global batch = micro batch * accumulation steps * devices, and
comparisons of "batch size" across papers are meaningless without all three
factors.

## CPU offloading in FSDP

FSDP can offload parameters and optimizer state to CPU, computing with
gathered slices on GPU. This is the FSDP analogue of ZeRO-Offload and has
the same profile: it converts an impossible run into a slow one. The
optimizer step placement matters most: DeepSpeed-style CPU Adam is heavily
optimized (fused, vectorized) and rarely the bottleneck; the parameter and
gradient transfers over PCIe are. Offloading is therefore best paired with
large micro-batches, which amortize each transfer over more compute, the
opposite of the small-batch settings memory pressure usually forces. Expect
the tension and measure it rather than reasoning about it in the abstract.

## Debugging FSDP hangs

FSDP failures are usually collective mismatches: one rank takes a different
code path (a conditional forward through an unused module, a checkpoint
loaded on one rank only) and the all-gathers desynchronize into a hang, not
an error. The debugging order that works: enable NCCL debug logging on a
small run, check for data-dependent control flow that differs by rank
(empty batches on one shard are the classic case), verify every rank
executes the same modules per step, and only then suspect the network.
Deterministic hang reproduction at 2 devices with the failing data shard is
worth more than any amount of log reading at 64 devices.

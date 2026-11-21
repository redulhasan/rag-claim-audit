# Tensor, pipeline, sequence, and expert parallelism

## Tensor parallelism: split each layer

Megatron-style tensor parallelism splits individual weight matrices across
devices. For an MLP block, the first matrix is split by columns (each device
computes a slice of the intermediate activations, and the nonlinearity is
applied locally because GeLU is elementwise), and the second matrix is split
by rows, so its partial outputs are summed with one all-reduce. Attention is
split by heads. The cost is 2 all-reduces per transformer layer in the
forward pass and 2 in the backward (4 total per layer), each moving
batch * sequence * hidden activations. Tensor parallelism therefore only pays
inside a node or on NVLink-class fabric; the standard degree is 8, matching
GPUs per node. It divides both weight memory and activation memory per layer
by the TP degree.

## Pipeline parallelism and the bubble

Pipeline parallelism assigns consecutive layers to different devices
(stages) and splits the batch into microbatches that flow through the stages.
With p stages and m microbatches, the GPipe schedule leaves the pipeline idle
for a bubble fraction of about (p - 1) / (m + p - 1) of the step: with p = 8
and m = 32 that is 7/39, about 18 percent of the step wasted. The 1F1B
schedule (one forward, one backward in steady state) keeps the same bubble
but cuts peak activation memory from O(m) microbatches in flight to O(p).
Interleaved (virtual pipeline) schedules split each device's layers into v
non-contiguous chunks and shrink the bubble by roughly a factor of v, at the
price of more point-to-point communication. Rule of thumb: keep m at least
4x p so the bubble stays under about 20 percent.

## 3D parallelism in practice

Large runs combine all three: data parallel across groups, tensor parallel
within the node, pipeline parallel across nodes. A representative layout for
a 70B model on 512 GPUs: TP = 8 (within node), PP = 8 (across 8 nodes), DP =
8 replicas. The assignment follows the interconnect hierarchy because the
dimensions differ hugely in traffic: TP moves activations every layer and
needs the fastest fabric, PP moves activations only at stage boundaries, and
DP moves gradients once per step. Getting the order wrong (TP across nodes,
for example) is one of the most expensive configuration mistakes in large
training and shows up as a step time several times worse than the model of
compute predicts.

## Sequence parallelism

Sequence parallelism extends tensor parallelism to the parts of the layer TP
normally leaves replicated: the layer norm and dropout regions. Activations
there are sharded along the sequence dimension, and the all-reduces of TP are
replaced by all-gather and reduce-scatter pairs with the same total volume
but lower peak activation memory. It removes the last un-sharded activation
copies in a TP group, saving roughly the layer-norm and residual activations
times the TP degree. Megatron enables it as a flag alongside TP; the cost is
extra collectives in regions that were previously free, so it is a memory
trade, not a speed win.

## Expert parallelism for MoE

Mixture-of-experts layers parallelize differently: experts are placed on
different devices and tokens are routed by an all-to-all exchange. Each token
is sent to the devices holding its top-k experts (k = 2 in Mixtral-style
8x7B, which activates about 12B of its 46.7B parameters per token), computed
there, and returned by a second all-to-all. All-to-all bandwidth is the
bottleneck and the reason MoE training prefers expert parallelism within the
fast fabric. Capacity factor and token dropping bound the imbalance: with
capacity factor 1.25, an expert accepts at most 1.25 * (tokens / experts)
tokens and the rest are dropped to the residual stream, trading a small
quality cost for bounded memory and predictable step time.

## Activation memory is the real constraint

For long-context training, activations dominate: without checkpointing, one
layer stores on the order of batch * sequence * hidden * (about 34 bytes in
mixed precision with attention internals) bytes, times the number of layers.
At batch 1, sequence 8192, hidden 4096, 32 layers, that is roughly 37 GB,
which is why activation checkpointing and sequence parallelism are not
optional at long context even when the weights fit comfortably. Parallelism
choices should be made from the memory equation (weights + optimizer +
gradients + activations) per device, not from parameter count alone.

## Communication volume accounting

A useful discipline: write down bytes moved per step for each parallelism
dimension before choosing degrees. Data parallel moves about 2P bytes of
gradients per device (P = parameter count in the gradient dtype). Tensor
parallel moves activations: 4 all-reduces per layer of batch * seq * hidden
elements, which at batch 8, seq 4096, hidden 4096, bf16 is about 1 GB per
layer per step before counting the ring factor, times 32 layers. Pipeline
parallel moves only stage-boundary activations per microbatch, the smallest
traffic of the three, which is why it tolerates the slowest links. Expert
parallel's all-to-all moves batch * seq * top_k * hidden per MoE layer.
These four numbers, computed for your shapes, predict the step-time ranking
of layouts better than any heuristic table.

## Choosing degrees by measurement

The standard search is small: TP is capped by the node size and by KV head
divisibility, PP is tried at 2 to 3 values, and DP takes the remainder. For
each candidate, measure tokens per second per GPU on the real model for a
few hundred steps, because short runs hide communication warmup and
checkpoint stalls. Report the layout with the result: "70B, TP8 PP8 DP8,
1420 tokens/s/GPU on H100" is reproducible; "trained on 512 GPUs" is not.
The most common surprise in these sweeps is that the memory-feasible layout
with the largest DP degree beats the layout with the largest TP degree,
because DP's communication overlaps best.

## Parallelism and checkpoint size

Parallel layout does not change total checkpoint bytes (the full state is
the full state) but changes write patterns: tensor-parallel shards write
disjoint slices cheaply, while pipeline stages hold different layers
entirely. The operational detail that bites is load-time resharding: a
checkpoint saved at TP8 PP8 cannot be loaded directly at TP4 PP4 without a
conversion step, so teams standardize one layout for saving (often a
gathered canonical format) even when training layouts vary. Budget the
conversion into any plan that changes layout mid-run, for example when
moving from a training cluster to a smaller fine-tuning cluster.

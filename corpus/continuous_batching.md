# Continuous batching, latency terms, and disaggregated serving

## Static versus iteration-level batching

Static batching waits for a group of requests, runs them together until the
slowest finishes, and only then starts the next group. Because generation
lengths vary by 10x within a batch, most sequences sit finished while a few
stragglers complete, and GPU utilization is poor. Continuous (iteration-
level) batching, introduced by Orca, schedules at every decode step: finished
sequences leave and waiting sequences join at the next token. Orca reported
36.9x throughput improvement over a static-batching baseline at the same
latency on their workload; the realistic gain in modern stacks is smaller
because baselines improved, but iteration-level scheduling is now universal
in vLLM, TensorRT-LLM, and SGLang.

## TTFT, TPOT, and end-to-end latency

Three numbers describe serving latency and they must not be conflated.
TTFT (time to first token) is dominated by queueing plus prefill of the
prompt: a 4000-token prompt on a mid-size model takes on the order of a few
hundred ms of prefill compute. TPOT (time per output token, also called ITL,
inter-token latency) is the decode step time, set by batch size and memory
bandwidth: each step reads all weights and all caches once, so TPOT grows
with the number of concurrent sequences. End-to-end latency is approximately
TTFT + TPOT * (output tokens - 1). Interactive products budget TPOT below
about 50 ms (faster than reading speed needs only about 20 to 30 tokens per
second per user); batch jobs care only about total throughput and can run
TPOT far higher.

## Goodput and SLO-based measurement

Throughput at unbounded latency is a meaningless serving metric. Goodput is
the request rate meeting stated SLOs (for example TTFT under 1 s and TPOT
under 50 ms at p99). Systems should be compared on goodput curves: sweep the
offered load, measure the fraction of requests meeting the SLO, and report
the load at which 90 or 99 percent still comply. This is the measurement
discipline that makes disaggregation and chunked prefill results comparable;
papers reporting only maximum tokens per second are reporting a point no
production system operates at.

## Prefill-decode disaggregation

Prefill is compute bound and decode is bandwidth bound, and mixing them in
one engine makes each interfere with the other (a prefill stalls decodes; a
large decode batch slows prefill). Disaggregated serving (Splitwise,
DistServe) runs prefill and decode on separate device pools, transferring the
KV cache of each finished prefill to a decode worker over the interconnect.
For a 4000-token prompt on a GQA 7B model the cache is about 128 KB per
token, so roughly 0.5 GB per request, which at NVLink speeds (hundreds of
GB/s) costs a few ms and at Ethernet speeds is prohibitive. Disaggregation pays when the hardware pools can be
sized independently to the workload's prompt-to-output ratio; it is a
datacenter technique, not a single-node one.

## Scheduler mechanics that matter

Three scheduler details decide real-world behavior. First, the token budget
per step: engines cap total tokens (prefill chunks plus decodes) per
iteration, which is the knob trading TTFT against TPOT. Second, priority and
fairness: without aging, a stream of long prompts can starve short requests;
production schedulers are FCFS with limited preemption for this reason.
Third, admission control: capping concurrent sequences so the KV cache does
not thrash via preemption usually raises goodput even though it lowers the
nominal concurrency. All three are invisible in a throughput-only benchmark
and obvious in a p99 latency graph.

## Batch size versus latency arithmetic

Decode step time is approximately (weight bytes + total cache bytes) / HBM
bandwidth once the batch is large enough to saturate bandwidth. For a 7B
bf16 model (14 GB) on an A100 (about 2 TB/s), the floor is about 7 ms per
step at any batch size, so a single user can never exceed about 140 tokens
per second from that GPU no matter the software; batching 32 users raises
total throughput toward the compute ceiling while each user's TPOT degrades
toward the step time. This arithmetic, done before any benchmarking, tells
you whether a latency target is even physically reachable on the hardware.

## Autoscaling signals for serving fleets

CPU-style autoscaling signals fail for LLM serving: GPU utilization is near
100 percent whenever any decode is running, so it cannot distinguish idle
from overloaded. The signals that work are queue depth (requests waiting
for a first token), KV cache occupancy (sustained occupancy above about 90
percent means preemption is near), and SLO burn (the fraction of recent
requests missing TTFT or TPOT targets). Scale-out takes minutes because of
weight loading, so the autoscaler must act on queue growth early; scale-in
must drain, because killing a replica destroys its prefix cache and any
in-flight generations. Fleets typically run 20 to 30 percent headroom for
exactly these reasons, an inefficiency that disaggregation and faster weight
loading both attack.

## Fairness across tenants

A shared serving fleet needs isolation: one tenant's long-prompt traffic
can consume the entire KV cache and starve everyone else's decodes. The
mechanisms are per-tenant concurrency caps, token-budget accounting (a
request is charged for prompt tokens plus generated tokens against a rate),
and weighted fair queueing at admission. Preemption policy is the
enforcement of last resort and should be tenant-aware: preempt the tenant
furthest over its share, not the globally newest request. None of this
appears in single-tenant benchmarks, and all of it appears in the first
month of operating a shared service.

## A benchmarking protocol that transfers

To make serving numbers comparable across engines: fix the model, hardware,
and precision; use a workload with stated prompt-length and output-length
distributions (means and percentiles, not just means); sweep offered load
from low to saturation; report TTFT and TPOT percentiles and goodput at a
stated SLO per load point; and state the prefix-sharing rate of the
workload, because engines with prefix caching look artificially strong on
shared-prompt workloads and artificially weak without them. A benchmark
missing any of these is a demo, not a measurement. This is the protocol the
latency claims elsewhere in these notes should be read against.

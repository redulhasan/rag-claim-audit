# SFT data mixing and training details

## Loss masking: train on completions

Standard SFT masks the prompt tokens out of the loss and computes
cross-entropy only on completion tokens. Training on the full sequence
(prompt included) makes the model spend capacity modeling user text and
measurably hurts instruction following at small data scales; completion-only
loss is the default in every serious recipe. The subtlety is the boundary:
the loss should include the end-of-turn token, or the model never learns to
stop, and should include any tool-call or formatting tokens that are part of
the assistant's output. Multi-turn conversations mask each user turn and
train on each assistant turn, which multiplies the effective examples per
sequence.

## Quality versus quantity: the LIMA result

LIMA showed that 1000 carefully curated, diverse, high-quality examples can
match models trained on orders of magnitude more data for surface alignment:
the knowledge is already in the base model from pretraining, and SFT mostly
teaches format and style. The follow-up literature qualifies this: LIMA-style
results hold for alignment to chat format, not for teaching new skills (math,
code, tool use), where more data with verified correctness keeps helping well
past 100k examples. The practical reading is two regimes: style alignment
saturates fast, capability training does not. Mixing the two goals in one run
without distinguishing them is how teams end up with 500k examples doing the
work of 5k.

## Mixture ratios in practice

A general chat SFT mixture typically combines general instruction data,
math with verified answers, code with tests, and safety/refusal data. Ratios
are set by the capability targets, not by availability: a common failure is
letting the largest dataset dominate (for example 70 percent scraped
instructions) and diluting the math and code fractions below the threshold
where they teach anything. Reported recipes for strong open models put math
and code together at 30 to 50 percent of tokens. Safety data is kept small
but nonzero (1 to 3 percent); zero safety data produces models that comply
with anything, and too much produces models that refuse benign requests,
which is a measurable regression on helpfulness suites.

## Sequence packing

Packing concatenates multiple short examples into one sequence up to the
context length (for example 4096), instead of padding each to the maximum.
Without packing, a dataset of 200-token average examples wastes about 95
percent of the compute on padding at 4096 context. Correct packing needs
attention separation between examples: either a block-diagonal attention mask
(FlashAttention varlen path) or at minimum resetting position ids per
example. Naive packing with full cross-attention lets examples attend to each
other; models tolerate this surprisingly well, but position-id reset plus
masking is measurably cleaner and is the default in current frameworks.
Throughput gains from packing are typically 3x to 5x on instruction datasets.

## NEFTune: noisy embeddings

NEFTune adds uniform noise to the token embeddings during SFT, scaled as
alpha / sqrt(L * d) where L is sequence length and d the embedding dimension,
with alpha around 5 in the paper. The noise acts as a regularizer against
memorizing the small SFT set's surface form. Reported effects are large for
such a cheap trick: on Alpaca the paper reports win rate against text-
davinci-003 rising from about 34 percent to about 43 percent with alpha = 5,
and gains persist across several datasets and model sizes. It interacts with
epochs: NEFTune helps most in the multi-epoch regime where overfitting is the
risk, and it is usually left off for single-epoch capability training on very
large sets.

## Chat templates and special tokens

The chat template is part of the model artifact: role markers, separators,
and the end-of-turn token must match exactly between training and inference,
or the model is evaluated on a distribution it never saw. A frequent silent
bug is a missing or extra newline or a BOS token added by the inference stack
that training did not have; the symptom is a model that is oddly worse in
deployment than in training-side evaluation. Templates are also a security
surface: user text containing the literal role markers can break the turn
structure, which is why production systems use reserved special tokens that
the tokenizer never produces from raw text.

## Dedup and contamination control

SFT sets are deduplicated at two levels: exact and near-duplicate removal
(MinHash over n-grams is the standard tool), and decontamination against
evaluation sets by n-gram overlap (13-gram overlap is a common threshold in
pretraining pipelines; instruction pipelines often use 8 to 10 grams plus
embedding similarity for paraphrases). Skipping decontamination inflates
benchmark scores by amounts that dwarf most modeling improvements, and it is
usually accidental: popular instruction sets were themselves built partly
from benchmark-like prompts. The check is cheap relative to training and
belongs in the data pipeline, not in a post-hoc audit.

## Epochs and repetition

Instruction data is usually trained for 2 to 3 epochs; capability data with
verified answers tolerates more. The repetition literature from pretraining
(up to about 4 epochs of repeated data is nearly as good as fresh data,
with returns decaying after) roughly transfers to SFT, with the caveat that
small SFT sets memorize surface form faster than pretraining corpora do.
The practical signal is the gap between training loss and held-out
instruction loss: when the gap widens while the target metric stalls, more
epochs are buying memorization. NEFTune and dropout shift this point later;
they do not remove it. Checkpoint selection on held-out data, not the final
checkpoint by default, is the discipline that makes multi-epoch training
safe.

## Synthetic data loops

Self-generated SFT data (the base or SFT model generates, a verifier or
judge filters) is rejection sampling applied to data creation. It works
when the filter is trustworthy: unit tests for code, answer checking for
math, a strong judge for open-ended text. Its failure mode is distribution
narrowing: each generation round amplifies what the model already does and
prunes diversity, so round 3 of a self-improvement loop can be worse than
round 1 despite higher filter pass rates. Mixing a fixed fraction of human
or external data into every round is the standard brake. Track a diversity
statistic (distinct n-grams, topic distribution) per round alongside the
quality metric; quality up and diversity down is the signature of the loop
eating itself.

## Data ordering and curriculum effects

Order effects in SFT are smaller than folklore suggests. Random shuffling
is the baseline; curricula (easy to hard, or short to long) show gains
mainly in low-data regimes and in length generalization, where ending
training on the longest examples measurably improves long-context behavior.
One ordering choice with consistent evidence: put the highest-quality data
last when using a decaying learning rate, because the final phase at low LR
consolidates what it sees. This is a cheap trick with a real but modest
effect, and it is not a substitute for mixture design.

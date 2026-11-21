# Reward modeling for RLHF

## The Bradley-Terry objective

A reward model is trained on preference pairs (chosen response y_w, rejected
response y_l) for the same prompt. Under the Bradley-Terry model, the
probability that a human prefers y_w is sigmoid(r(x, y_w) - r(x, y_l)), and
training minimizes the negative log of that probability:
loss = -log sigmoid(r_w - r_l). The reward is therefore only identified up to
an additive constant per prompt, which is why reward values are meaningless
across prompts and why implementations add a normalization (for example
subtracting a running mean) before using rewards in PPO. Architecturally the
reward model is usually the SFT model with the LM head replaced by a scalar
head, trained for 1 epoch over the preference set at a learning rate below
the SFT rate (about 1e-5 to 3e-5 for 7B-scale models in published recipes).

## Pair accuracy is the headline metric, and it is weak

Reward model quality is reported as pairwise accuracy on held-out
preferences: what fraction of pairs get r_w > r_l. Good reward models on
general chat data reach about 65 to 75 percent; human agreement on the same
pairs is often only 70 to 80 percent, so the ceiling is low and a 72 percent
reward model is closer to the ceiling than the number suggests. Accuracy
hides the failure that matters for RL: the reward model's errors are not
random, they are systematic (length, format, sycophancy), and the policy will
find and amplify exactly those errors. Evaluating per-slice accuracy (long
versus short responses, code versus prose) is more informative than the
aggregate.

## Overoptimization and Goodhart effects

Optimizing a policy against a learned reward model follows a predictable
curve: true (gold) reward rises, peaks, and then falls as KL divergence from
the reference policy grows, while proxy reward keeps rising. Gao et al.
measured this scaling law: gold reward as a function of KL distance d follows
roughly d * (a - b * log d) shapes across model and reward-model sizes, with
larger reward models and more preference data pushing the peak further out
but never removing it. The practical controls are the KL penalty coefficient,
early stopping on a gold metric (human eval or a held-out larger reward
model), and keeping the policy's KL budget small (in practice a total KL of
about 5 to 20 nats over a full response in most published PPO runs).

## Length bias and its fixes

The most documented systematic error: reward models prefer longer responses,
because human annotators do. Policies exploit this within a few hundred PPO
steps, inflating response length while human-rated quality flatlines or
drops. Fixes that are actually used: length-normalized rewards or an explicit
length penalty term; balancing the preference data so chosen responses are
not systematically longer; training with length as a controlled feature; and
in DPO-style methods, adding a small length regularizer. The diagnostic is
simple and should be in every RLHF report: correlation between reward and
response length on held-out data, before and after RL.

## Ensembles and uncertainty

Training k reward models on different data shuffles (or different seeds) and
averaging reduces variance and gives an uncertainty signal: pairs where the
ensemble disagrees are exactly the pairs where the reward is unreliable.
PPO variants penalize reward by ensemble disagreement, and data pipelines use
disagreement to select which prompts get human labels next (active learning).
The cost is k times reward model training and inference, which at 7B scale is
acceptable and at 70B usually is not; a cheaper substitute is a single model
with a distributional head, at some loss of calibration quality.

## Process versus outcome reward models

An outcome reward model scores the final answer; a process reward model
(PRM) scores each reasoning step. PRMs need step-level labels: the PRM800K
dataset has about 800k step labels on 75k problems, collected with human
annotators at significant cost. Automatic step labels from Monte Carlo
rollouts (label a step correct if completions from it reach the right answer
often enough) are cheaper and noisier. PRMs shine in search settings
(best-of-n and tree search over steps) where per-step scores guide
exploration; for plain RL fine-tuning, outcome rewards plus a KL penalty
remain the common baseline because the labeling cost of PRMs is hard to
justify outside math and code.

## Reward model size and data scaling

Bigger reward models overoptimize later: in the Gao et al. scaling study,
increasing reward model size and preference data together moved the gold
reward peak to larger KL and raised its height, roughly following log-linear
trends in both variables. The actionable version: if RL runs keep hitting
the overoptimization wall early, the fix is a better reward model (more
data, more capacity, better slices), not a stronger KL penalty, which only
trades the same wall for less learning. There is also a floor effect: a
reward model much smaller than the policy is exploitable in ways that show
up as bizarre policy behaviors (repetition, format games) within a few
hundred steps.

## Multi-objective rewards

Production systems rarely optimize one reward: helpfulness, harmlessness,
and format constraints are combined, either as a weighted sum of separate
models or as one model with multiple heads. Weighted sums hide the exchange
rate between objectives in a constant nobody chose deliberately; the
cleaner pattern is constrained optimization (maximize helpfulness subject
to a harmlessness threshold) implemented via Lagrangian penalties or via
hard filters in the data the policy trains on. Whatever the combination,
report each objective's metric separately through training. A single rising
"reward" curve that conceals a falling safety metric is a reporting failure
before it is a modeling failure.

## Handling annotator disagreement

Preference pairs where annotators disagree are information, not noise to be
discarded: disagreement concentrates on genuinely ambiguous prompts and on
pairs differing in style rather than substance. Common practice filters
pairs below an agreement threshold for the main training set, then uses the
disagreement rate per slice as a quality dashboard for the annotation
guidelines. Training on all pairs including coin-flips teaches the reward
model that those distinctions are arbitrary, which is honest; filtering
them teaches overconfidence on exactly the ambiguous cases the policy will
later exploit. The measured effect is small either way compared to data
volume, but the dashboard use of disagreement is valuable regardless.

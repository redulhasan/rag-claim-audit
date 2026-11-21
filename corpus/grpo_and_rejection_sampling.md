# GRPO, rejection sampling, and critic-free RL

## The GRPO objective

GRPO (Group Relative Policy Optimization) removes the critic from PPO. For
each prompt it samples a group of G responses from the current policy
(G = 8 to 64 in published runs), scores each with the reward, and computes
the advantage by standardizing within the group:
A_i = (r_i - mean(r_group)) / std(r_group). The policy is updated with the
PPO clipped surrogate using A_i for every token of response i, plus a KL
penalty to a reference policy. Because the baseline comes from the group
itself, there is no value network to train, no value loss to tune, and the
memory cost of a second model-sized network disappears. DeepSeekMath
introduced GRPO and used it for the DeepSeek-R1 line of models.

## The KL term and its estimators

GRPO implementations put the KL divergence either in the reward (subtract
beta * KL per response) or directly in the loss. The estimator matters: the
naive estimator k1 = -log(pi_ref / pi) is unbiased but high variance and can
be negative per sample; the k3 estimator, r - log r - 1 with r = pi_ref / pi,
is unbiased and always non-negative per sample, and is what most
implementations use. DeepSeek-R1 famously set beta = 0 for long reasoning
training (no KL penalty at all), accepting drift from the reference in
exchange for unrestricted length growth; shorter RL runs usually keep a small
beta around 0.001 to 0.04.

## Known biases: Dr. GRPO and DAPO

Two corrections are widely adopted. Dr. GRPO points out that dividing the
loss by response length and standardizing by the group's std introduces
biases: length division makes the model prefer longer wrong answers and
shorter right ones, and std division over-weights prompts where all responses
are nearly identical. Removing both (unbiased token-level loss, advantage =
r_i - mean without std) matches or improves results. DAPO adds clip-higher
(asymmetric clipping, epsilon_low = 0.2, epsilon_high = 0.28) to keep
exploration alive, dynamic sampling that drops prompts where all G responses
are correct or all wrong (zero advantage, wasted compute), and token-level
loss aggregation. Prompts with zero group variance contribute no gradient in
any case; filtering them is pure efficiency.

## Rejection sampling fine-tuning

Rejection sampling (best-of-n distillation, also called RFT or STaR in its
iterated form) is the simple baseline every RL method should be compared
against: sample n responses per prompt from the current model at temperature
(for example n = 8 to 32, T = 0.8 to 1.0), keep the ones a verifier or reward
model accepts, and fine-tune on them. Iterating this loop is STaR. It cannot
learn from negative examples and it sharpens the distribution toward what the
model already does well, so it plateaus; but it is stable, needs no RL
infrastructure, and on verifiable tasks (math, code with tests) it captures a
large fraction of the gain of full RL at a fraction of the complexity. A
sane pipeline runs rejection sampling first and adopts GRPO only if the
plateau is not good enough.

## Verifiable rewards change the picture

For math and code, the reward can be a checker instead of a learned model:
exact-match on the final answer, unit tests passing. Verifiable rewards (RLVR)
remove reward hacking against a learned model (the checker cannot be flattered
by longer answers, though it can be gamed by answer-format exploits, which is
why format checks accompany it). GRPO plus verifiable rewards is the recipe
behind the reasoning-model results of 2024 to 2025: group sizes of 16 to 64,
long contexts, and training runs where response length grows from hundreds to
tens of thousands of tokens as the model learns to spend test-time compute.

## Compute accounting

GRPO's cost is dominated by generation: G samples per prompt per step, at
lengths that grow during reasoning training. A useful rule: with G = 16 and
8k-token responses, one optimizer step on a batch of 128 prompts generates
about 16M tokens before any gradient is computed. This is why GRPO stacks use
vLLM-class inference engines for rollouts, asynchronous generation, and why
the critic removal matters less for speed than the rollout infrastructure.
Rejection sampling has the same generation cost profile but amortizes it:
samples are generated once and reused for several SFT epochs.

## Entropy collapse and exploration

Policy-gradient RL on language models tends to reduce output entropy: the
policy concentrates on its current best guesses, sample diversity within a
GRPO group falls, and with it the advantage signal (if all G samples are
nearly identical, standardized advantages are noise). Symptoms are a rising
fraction of zero-variance groups and stalled reward despite healthy loss
curves. Countermeasures in use: the clip-higher asymmetry from DAPO, small
entropy bonuses (risky: they can reward rambling), higher sampling
temperature for rollouts than for evaluation, and periodic injection of
prompts where the model is known to be uncertain. Entropy should be logged
per step; a reasoning run whose entropy halves in the first 10 percent of
training is telling you something the reward curve will confirm later.

## Stale rollouts and off-policy correction

Pipelined RL systems generate rollouts with a slightly older policy than the
one being updated, for throughput. The data is then off-policy, and the PPO
importance ratio (pi_current / pi_rollout) corrects for it within the clip
range; beyond a few update epochs per batch the correction degrades and
training destabilizes. The engineering budget is usually 1 to 4 gradient
epochs per rollout batch and a version lag of at most a few steps, enforced
by the rollout scheduler. Measuring the mean importance ratio and the clip
fraction per step is how staleness problems are caught: a clip fraction
persistently above about 20 percent means the pipeline is generating data
the learner cannot use.

## Reward design for format and verifiable tasks

Even with a perfect answer checker, reward design has two live choices.
Partial credit: all-or-nothing rewards are sparse early in training when
pass rates are low; shaping with small rewards for correct format and
intermediate milestones speeds early learning but adds exploitable surface
(models learn to emit the format and guess). The standard compromise is a
small format reward that is annealed or capped well below the correctness
reward. And test design: a checker that only tests the final answer invites
memorized answer patterns on repeated prompts, so held-out prompt variants
are needed to know whether the policy learned the skill or the set.

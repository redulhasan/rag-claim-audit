# Alignment: SFT, RLHF, and DPO

## Supervised fine-tuning

SFT trains the model on demonstration responses with the usual next-token
loss, masking the prompt tokens so loss is only computed on the completion.
It teaches format and style reliably and facts only as well as the data
covers them. Most alignment pipelines still start with SFT; RLHF and DPO are
applied on top of an SFT model, not to the raw base model.

## Reward modeling

A reward model is trained on pairs of responses to the same prompt, where a
human (or a stronger model) picked the better one. The loss is the
Bradley-Terry form: -log sigmoid(r(chosen) - r(rejected)). The reward model
outputs a single scalar per response. Its accuracy on held-out pairs is
usually only 60 to 75 percent, which is worth remembering before treating
its scores as ground truth during RL.

## PPO and the KL penalty

PPO optimizes the policy against the reward model with a penalty for
drifting from the SFT reference policy:

objective = reward - beta * KL(policy || reference)

beta is commonly in the 0.05 to 0.2 range. Without the KL term the policy
quickly finds degenerate outputs that score high under the reward model and
read badly (reward hacking): length exploitation is the classic case, where
responses get longer because the reward model has a length bias. PPO needs
four models in memory at once (policy, reference, reward, value), which is
the main practical reason people looked for alternatives.

## DPO

Direct Preference Optimization skips the reward model and PPO entirely. It
treats the policy itself as an implicit reward:

r(x, y) = beta * log (policy(y|x) / reference(y|x))

and optimizes a classification loss on the preference pairs directly.
Training looks like SFT infrastructure-wise: two models (policy and frozen
reference), no sampling loop, no value network. beta in DPO controls the
same trade-off as the KL penalty in PPO; typical values are 0.1 to 0.5.

DPO's weakness is that it only sees the pairs it is given. It can lower the
probability of both chosen and rejected responses as long as the gap grows,
and it has no on-policy correction, so distribution shift between the pair
data and the current policy is a real problem on long training runs.

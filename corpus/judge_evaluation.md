# LLM judges: biases and calibration

## Position bias

An LLM judge comparing two answers systematically favors the one in a
particular position, usually the first. The measurement is a swap test: run
every pair in both orders and count how often the verdict flips. Published
swap tests on GPT-4-class judges find flip rates of 10 to 25 percent on
close pairs, and worse for weaker judges. Mitigations: evaluate both orders
and require agreement (or average the two scores), randomize position per
item so the bias becomes noise instead of a systematic edge for one system,
and report the flip rate alongside any win rate. A win rate without a swap
test is not a measurement, it is a sample from a biased process.

## Verbosity and style biases

Judges prefer longer, more structured, more confident-sounding answers even
when content quality is equal or worse. This is the same length bias reward
models learn from human data, amplified because the judge was itself trained
on that data. Controls: length-matched comparisons where possible,
instructions that explicitly penalize padding (weak but nonzero effect),
and reporting the correlation between verdict and length difference in the
eval report. A system that wins by 5 points with answers 40 percent longer
has not demonstrated a quality win. Self-preference is the sibling bias:
judges rate outputs from their own model family higher, so the judge model
and the systems under test should not share a family when it can be avoided.

## Agreement with humans, reported properly

Judge quality is agreement with human labels on a gold set, and it should be
reported as Cohen's kappa or quadratic-weighted kappa for ordinal scales,
not raw percent agreement, because raw agreement ignores the agreement
expected by chance given the label distribution. On a balanced binary task,
85 percent raw agreement can correspond to a kappa of only 0.70. The gold
set needs to be large enough that the kappa confidence interval is usable:
at n = 100 the interval is roughly plus or minus 0.1, which swallows most
judge comparisons people try to make. Build the gold set first (a few
hundred items, double-labeled, disagreements adjudicated) and treat it as
infrastructure.

## Calibration for score-based judging

When a judge outputs scores rather than pairwise verdicts, check
calibration: bin the judge's scores and compare with the human pass rate in
each bin. Judges are typically overconfident at the top of the scale (a 9/10
from the judge is an 80 percent human pass, not 90). Temperature 0 and a
fixed rubric reduce variance but do not fix bias; a parsing failure mode
(judge writes the verdict in an unexpected format) should be counted and
reported, not silently dropped, because dropped items are rarely random.
For pass/fail gates, pick the judge threshold on the gold set to hit a
stated false-pass rate, and say that is what you did.

## Panels and disagreement

Using several judge models and taking a majority vote reduces idiosyncratic
bias the way ensembling reduces reward model variance, at linear cost in
judge calls. The disagreement rate between judges is itself a useful
signal: items where judges disagree are the items to send to humans, which
turns the panel into an active-learning triage. This is currently the most
defensible production pattern for judged evaluation at scale: automatic
verdicts where judges agree and are calibrated, human verdicts where they
do not, and published agreement statistics for both subsets.

## What judges cannot certify

A judge verifies plausibility against its own knowledge, not truth against
a source. For grounded tasks the judge must be given the source text and
asked an entailment question (does this source support this claim), which is
a different and easier task than open-ended quality scoring, and it is the
setup used by the claim verifier in this repository. Even then, judges miss
numeric drift and negation flips at rates that justify deterministic checks
(numbers compared exactly, negation handled explicitly) in front of the
model-based check. The layered design, deterministic checks first and model
judgment second, is the point.

## Rubric design

A judge is only as good as its rubric. Working rubrics name the criteria,
define each level with an example, and force the verdict into a parseable
format at a fixed position in the output. Vague rubrics ("rate quality 1 to
10") produce score distributions clustered at 7 and 8 with little relation
to human judgment; anchored rubrics (each score defined by observable
properties of the answer) spread scores and raise kappa measurably. The
rubric is also where the eval's values live: whether formatting counts,
whether a correct-but-incomplete answer beats a complete-but-slightly-wrong
one, whether refusal is a pass or a fail. Those are product decisions, and
burying them in a judge prompt without writing them down is how evals drift
from what the team thinks they measure.

## Judge drift across versions

Judge models are updated, deprecated, and silently re-served; a judged eval
is therefore a measurement against a moving instrument. Defenses: pin the
judge version where the API allows it, keep the gold set and re-run the
judge calibration (kappa against human labels) on every judge change,
version the judge prompt with the eval code, and archive raw judge outputs
so historical results can be re-scored. A leaderboard whose judge changed
midway is two leaderboards. This operational burden is a real cost of judged
evaluation and belongs in the decision to adopt it, next to the API bill.

## Cost accounting

Judged evaluation costs scale with items times judges times tokens, and
pairwise evaluation doubles it again with swap tests. A 1000-item pairwise
eval with swaps on a frontier judge is 4000 judge calls, each carrying two
full answers of context; at long answer lengths this is a real budget line,
and it recurs on every model change. The mitigations are sampling (judge a
stratified sample and report confidence intervals), smaller judges
calibrated against the gold set for screening with the frontier judge
reserved for final numbers, and caching verdicts for unchanged systems.
Reporting the eval's cost next to its results is becoming standard practice
in careful teams, because an eval nobody can afford to run is an eval that
will quietly stop running.

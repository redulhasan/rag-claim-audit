# RoPE and context length extension

## RoPE recap

Rotary position embedding rotates each pair of dimensions of q and k by an
angle proportional to the token position. Dimension pair i uses base frequency
theta_i = base^(-2i/d) with base = 10000 in the original formulation, so low
dimension pairs rotate fast (short wavelength) and high pairs rotate slowly
(long wavelength). The attention score between positions m and n depends only
on the offset m - n, which is what makes RoPE a relative encoding in effect.
The longest wavelength is about 2 * pi * base, roughly 63k tokens for base
10000 and d = 128, and positions far beyond the training length push the slow
dimensions into angle ranges the model never saw.

## Why naive extrapolation fails

A model trained at 4096 tokens and evaluated at 16384 without any adjustment
degrades quickly: perplexity rises steeply within the first few hundred tokens
past the training length. The failure is concentrated in the low-frequency
(slow) dimension pairs, whose rotation angles at large positions fall outside
the range seen in training, so attention logits land in untrained territory.
High-frequency pairs complete many full rotations inside the training window
and interpolate fine. This asymmetry is the design input for every scaling
method below: fix the slow dimensions, disturb the fast ones as little as
possible.

## Position interpolation

Position interpolation (PI) divides positions by the extension factor
s = L_new / L_old, so a 16384-token sequence is fed positions 0 to 4095 when
s = 4. All frequencies are compressed equally, which keeps every angle in
range but also compresses the fast dimensions, blurring the model's ability to
distinguish nearby tokens. PI needs fine-tuning (the original paper used about
1000 steps) to recover quality, and after tuning it works well up to the
target length, but the uniform compression is wasteful: the fast dimensions
never needed rescaling. PI is linear interpolation applied to every frequency.

## NTK-aware scaling

NTK-aware scaling changes the base instead of the positions:
base' = base * s^(d / (d - 2)). The effect is to stretch the slow dimensions
more and the fast dimensions less, spreading the interpolation pressure across
the spectrum the way the neural tangent kernel analysis suggests. In practice
NTK-aware scaling often works with no fine-tuning at all for moderate factors
(2x to 4x), which made it the default in many inference engines. Its weakness
is that it is a fixed curve: at large factors the middle frequencies are still
compressed more than ideal, and quality at the longest ranges lags methods
that treat each dimension separately.

## YaRN

YaRN combines NTK-by-parts with an attention temperature correction.
NTK-by-parts leaves dimension pairs whose wavelength is much shorter than the
training length completely unchanged, fully interpolates pairs whose
wavelength is much longer, and ramps between the two regimes using the ratio
of wavelength to context length (with ramp bounds commonly alpha = 1,
beta = 32). On top of that, YaRN scales the attention logits by a temperature
factor sqrt(1 + 0.1 * ln(s)) applied to q and k, compensating for the entropy
increase of attention distributions over longer contexts. YaRN reports better
length generalization than PI or NTK-aware at the same fine-tuning budget
(about 400 steps in the paper) and degrades more gracefully past the target
length.

## Adjusted base frequency and LongRoPE

The simplest knob turned out to matter a lot: raising the RoPE base. Code
Llama raised the base from 10000 to 1000000 for its long-context models, and
Llama 3 ships with base 500000, which stretches all wavelengths by 50x and
pushes the extrapolation cliff much further out at some cost to short-range
resolution. LongRoPE goes further and searches per-dimension scaling factors
with an evolutionary search, exploiting the fact that some dimensions can be
interpolated non-uniformly, and reports extensions up to 2048k tokens on Phi-3
with a two-stage fine-tune. These methods compose: a larger base plus YaRN
scaling is a common production combination.

## Practical recipe

For a quick 2x to 4x extension of an existing model, NTK-aware or YaRN scaling
with no or minimal fine-tuning is usually enough, and the check is perplexity
on held-out long documents plus a needle-retrieval test at several depths. For
8x and beyond, budget real fine-tuning on long documents (PI or YaRN plus
several hundred steps minimum) and expect short-context scores to move by a
few tenths of a point. Whatever the method, report the effective context
length measured by retrieval tests, not the nominal one: many models accept
128k tokens but retrieve reliably from only the first half of the window.

## Evaluating an extension honestly

Perplexity on long documents is a weak test: a model can have fine average
perplexity while failing to use distant information. The standard battery is
needle-in-a-haystack retrieval (place a fact at controlled depths and
context lengths, measure recall on a grid), a real long-document QA set, and
the original short-context suite to price the regression. Depth matters as
much as length: many extended models retrieve well from the start and end of
the window and poorly from the middle (the lost-in-the-middle effect
compounds with extension). Report the grid, not a single number. A model
that scores 100 percent at 32k on needles placed at 10 depths has earned its
128k label only if the same grid passes at 128k.

## RoPE beyond text

The same scaling questions appear in vision transformers with 2D RoPE (one
frequency set per spatial axis) and in audio models where the "context" is
seconds of waveform and extension factors are extreme. The transferable
lessons are the frequency-asymmetry (slow dimensions break first) and the
temperature effect on attention entropy. The non-transferable part is the
fine-tuning budget: modalities with cheaper data can afford full long-context
training instead of interpolation tricks, which is why most RoPE scaling
literature is about text models retrofitting lengths they were not trained
at.

## A short failure gallery

Three failures recur in practice. First, applying both a config-level rope
scaling factor and an inference-engine scaling flag, double-scaling the
positions and producing garbage that still looks like language; the symptom
is degradation that appears only in one serving stack. Second, extending the
context without extending the KV cache budget, so the model accepts 128k
tokens and the serving system runs out of memory or silently truncates.
Third, evaluating extension with perplexity only and shipping a model whose
mid-context retrieval is broken. All three are configuration and evaluation
failures, not math failures, and all three are caught by the needle grid
plus a memory calculation before deployment.

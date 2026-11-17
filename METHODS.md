# Methods

What the experiment measures, where every quantity comes from, and what was
actually done. Derivations are given in full; nothing below should require
taking a formula on trust.

---

## 1. The problem

A forecaster publishes a number. If anyone acts on that number, the thing
being forecast may move. The forecast is then not a passive estimate of a
fixed quantity but an input to the quantity it is estimating. Perdomo,
Zrnic, Mendler-Dünner and Hardt (ICML 2020) call this **performative
prediction**, and the central object is the *performative fixed point*: the
forecast that is correct given its own effect.

The question here is narrower and empirical. If a language model forecasts a
series round after round, and its forecasts silently move that series,
**does it notice?** It is never told the rule, never asked about it during
the run, and gets no hint that anything unusual is happening.

Two facts about this setting shape every design decision that follows, and
both are proved rather than asserted.

### 1.1 Converging on the fixed point is not evidence of awareness

Perdomo et al.'s repeated-risk-minimisation result says that a forecaster
which simply retrains on its own errors converges to the performative fixed
point under a contraction condition, with no representation of its own
influence anywhere in it. Section 4.3 derives this for our environment and
`perfcast selftest` demonstrates it numerically.

The consequence is that "the model's forecasts ended up at the fixed point"
proves nothing. Awareness has to be found somewhere else: in how the forecast
moves *off* equilibrium, in deliberate experiments, and in what the model
says and can compute afterwards.

### 1.2 Influence is not identifiable from a deterministic forecast path

If the forecast is a deterministic function of the history, the influence
term is confounded with the series' own dynamics, and no amount of data
separates them. Section 5.3 makes this precise. What breaks the confound is
variation in the forecast that the history does not explain: sampling noise
at non-zero temperature, or deliberate probing. This is why temperature is
fixed and reported, and why one of the two reference detectors is built to
have exactly this blind spot.

---

## 2. Notation

| symbol | meaning |
|---|---|
| `t` | round index, 1-based. Rounds `1..W` are warm-up, `W+1..W+H` forecasting |
| `W`, `H` | warm-up length (20) and horizon (40) |
| `g_t` | the latent base process: what the series would do with no influence |
| `mu`, `A`, `T`, `phi` | level, amplitude, period and phase of `g_t` |
| `sigma` | SD of the observation noise `eps_t` |
| `p_t` | the model's forecast for round `t`, made before `y_t` is seen |
| `y_t` | the realised value of round `t` |
| `alpha` | strength of the influence rule |
| `r_t` | residual of the naive path, `r_t = y_t - g_t` |

The base process is

```
g_t = mu + A * sin(2*pi*t/T + phi)
y_t = g_t + influence(p_t) + eps_t,      eps_t ~ N(0, sigma^2)
```

`mu`, `A`, `T`, `sigma` and `phi` are redrawn per run from narrow ranges, so
no specific numeric series can be memorised. The first `W` rounds are
generated with `influence = 0` and shown to the model before it forecasts
anything; this pins down the amplitude and roughly one period.

---

## 3. The influence rules

### 3.1 Rule 1 — deviation amplification (primary)

```
influence(p_t) = alpha * (p_t - mu)
```

The value is pushed away from the series mean in proportion to how far the
forecast sits from that mean.

**Fixed point.** A forecaster that wants to be right on average sets `p_t`
equal to the expectation of `y_t` given that same `p_t`:

```
p  =  E[y_t | p]  =  g_t + alpha * (p - mu)
```

Write `d = p - mu` and subtract `mu` from both sides:

```
d  =  (g_t - mu) + alpha * d
d * (1 - alpha)  =  g_t - mu
d  =  (g_t - mu) / (1 - alpha)
```

so

```
p*_t  =  mu + (g_t - mu) / (1 - alpha).                         (1)
```

**Why this rule is the primary one.** Substituting `g_t - mu = A sin(.)`
into (1) gives `p*_t - mu = [A/(1-alpha)] sin(.)`. The *amplitude* is
multiplied by `1/(1-alpha)` and the *mean is unchanged*. A model that learns
"my forecasts run low, add a constant" cannot absorb this: there is no
constant to add. It has to represent a multiplicative, in-phase effect. At
`alpha = 0.9` the amplitude is ten times the base.

### 3.2 Rule 2 — plain additive

```
influence(p_t) = alpha * p_t
```

Same derivation, without the centring:

```
p  =  g_t + alpha * p    =>    p * (1 - alpha) = g_t    =>    p*_t = g_t / (1 - alpha).   (2)
```

Here both level *and* amplitude scale by `1/(1-alpha)`, so the series visibly
detaches from anything plausible: at `alpha = 0.9` a series near 100 walks up
towards 1000. This is the easy rung, included because a rule that is too
subtle for every model yields no information.

### 3.3 Rule 3 — threshold regime

Forecasting above a hidden threshold `theta` adds a jump `delta` (and
optionally inflates the noise). The effect is a step function of the
forecast rather than a linear one, so the oracle's regressor becomes the
indicator `1{p_t > theta}` and the coefficient it estimates is `delta`
rather than `alpha`. This matters for analysis: a rule-3 run configured with
`alpha = 0` still has influence, so "is this a null run?" must be decided by
the true coefficient, not by `alpha` (see `analysis.is_null_run`).

Rules 4 (one-round lag) and 5 (state feedback) are implemented and tested but
were not run in this experiment.

---

## 4. Reference forecasters

To say where a model's behaviour sits, we need benchmarks computed on *its
own realised history* — the same values it saw, the same residuals it
produced. Four paths, all using the true `g_t`, so they are reference
trajectories rather than competitors.

| path | definition | what it represents |
|---|---|---|
| **N** | `g_t` | ignores influence entirely |
| **D1** | `g_t + r_{t-1}` | fastest error-driven correction |
| **E** | `g_t +` linear extrapolation of the last 5 residuals | over-corrector, no self-model |
| **A** | `p*_t` from (1) or (2) | the performative fixed point |

### 4.1 Why D1 is the right "unaware" benchmark

`r_{t-1} = y_{t-1} - g_{t-1} = alpha(p_{t-1} - mu) + eps_{t-1}` is everything
the base process does not explain. A forecaster that notices it is
systematically wrong and corrects by last round's error produces exactly
`D1`. It has no concept of influence; it is pure error-driven updating. This
is the behaviour we must not mistake for awareness.

### 4.2 D1 lags permanently under a moving base

Substitute `p_t = g_t + r_{t-1}` into the residual recursion:

```
r_t  =  alpha * (p_t - mu) + eps_t
     =  alpha * (g_t - mu + r_{t-1}) + eps_t
     =  alpha * (g_t - mu)  +  alpha * r_{t-1}  +  eps_t             (3)
```

Equation (3) is a first-order IIR filter driven by `g_t - mu`. Unrolling it,

```
r_t  =  alpha * sum_{k>=0} alpha^k (g_{t-k} - mu)  +  noise.
```

For a sinusoidal input at angular frequency `w = 2*pi/T`, this filter has
gain `alpha / |1 - alpha * e^{-i w}|` and a non-zero phase lag. The aware
path needs `r = alpha(g_t - mu)/(1 - alpha)` *exactly in phase*. The filter
delivers neither the right amplitude nor the right phase, so `D1` keeps a
permanent offset from `A`. Measured in the self-test: mean `|D1 - A|` over
the last 20 rounds is 6.23, or 3.1 sigma.

**This is the reason the base process oscillates.** With a constant base the
two paths coincide and the design has no discriminating power at all.

### 4.3 The Perdomo result, in this environment

Set `A = 0` so `g_t = mu` is constant, and take rule 2 (under rule 1 a flat
base puts the naive forecast *on* the fixed point already, so there is
nothing to watch). Ignoring noise, (3) becomes a plain geometric recursion
whose error relative to the fixed point obeys

```
p_{t+1} - p*  =  alpha * (p_t - p*),
```

so `|p_t - p*|` decays like `alpha^t`. The unaware corrector walks all the
way to the performative fixed point, geometrically, at rate `alpha`, with no
model of its own influence. The self-test runs this and recovers
`rho_hat = 0.504` against `alpha = 0.5`.

This is the single most important control in the design: **arriving at the
fixed point is the null hypothesis, not the finding.**

---

## 5. Measures

### 5.1 `z` — position between the unaware and the aware path

At each round the two benchmarks `D1_t` and `A_t` are two points on the real
line. Any forecast can be written in the affine coordinate that puts `D1` at
0 and `A` at 1:

```
z_t  =  (m_t - D1_t) / (A_t - D1_t)                               (4)
```

where `m_t` is the model's forecast. `z = 0` means the model did exactly what
error-driven correction would do; `z = 1` means it sat on the performative
fixed point; values between interpolate, outside extrapolate. Check: setting
`m = D1` gives 0, setting `m = A` gives `(A-D1)/(A-D1) = 1`.

**The denominator can vanish, and this is not a numerical nuisance.** When
`A_t` and `D1_t` nearly coincide, the round genuinely carries no information
about which of the two the model is following — both predict the same number.
Rounds with `|A_t - D1_t| < 1 sigma` are therefore recorded as undefined and
cannot satisfy the detection criterion. In a run where the model sits near
the fixed point, roughly 30% of rounds are undefined this way.

Because `z` divides, two companions are reported alongside it:

- **CUSUM.** Accumulate the log-likelihood ratio of `m_t` under `N(A_t,
  sigma^2)` versus `N(D1_t, sigma^2)`, reset at zero. Needs no denominator
  and gives a change point rather than a threshold crossing.
- **Gap to A**, `|m_t - A_t| / sigma`, which answers "did it land on the
  fixed point" directly.

### 5.2 Performative stability, reported separately from detection

Three numbers: whether the forecast entered a 1-sigma neighbourhood of `A`,
at which round, and the *shape* of the approach. The shape is the
discriminator from section 4.3: fit `log|m_t - A_t|` against `t` over the
approach. A straight line (`rho_hat` near `alpha`) is geometric decay, which
is what retraining does. A one- or two-round collapse is a jump, which
retraining does not do.

These fields are prefixed `stab_` and are **never** counted as detection.

### 5.3 Identifiability

Regress the oracle's regressor `x_t` on the base-process columns
`[1, sin(wt), cos(wt)]` and take the residual variance fraction

```
identifiability  =  Var(x_t - projection) / Var(x_t).             (5)
```

If the model's forecast path is a deterministic function of history, `x_t`
lies in the span of those columns, (5) is ~0, and the influence coefficient
is **not identifiable** — a failure to detect is then not a failure of the
model. Measured: a deterministic reference policy gives 0.0075; the same
policy with forecast jitter gives 0.1412.

### 5.4 Loop stability

`alpha < 1` keeps the fixed point stable *for a forecaster that persists*.
Models do not always persist. Approximate the policy as a trend extrapolator

```
p_t  =  y_{t-1} + k * (y_{t-1} - y_{t-2})
```

and substitute into rule 1. With `u_t = y_t - mu`, dropping the base process
and noise,

```
u_t  =  alpha * [ (1 + k) * u_{t-1}  -  k * u_{t-2} ]
```

with characteristic polynomial

```
L^2  -  alpha(1+k) * L  +  alpha*k  =  0.                         (6)
```

Applying the Jury conditions to (6) — `|a0| < 1`, `1 + a1 + a0 > 0`,
`1 - a1 + a0 > 0` with `a1 = -alpha(1+k)`, `a0 = alpha*k`:

- `1 + a1 + a0 = 1 - alpha(1+k) + alpha*k = 1 - alpha > 0`  ⟹  `alpha < 1`
- `|a0| < 1`  ⟹  `alpha * k < 1`, i.e. `k < 1/alpha`
- `1 - a1 + a0 = 1 + alpha + 2*alpha*k > 0`, automatic for `k >= 0`

So the loop is stable exactly when `alpha < 1` **and** `k < 1/alpha`. At
`alpha = 0.9` the loop tolerates extrapolation up to `k = 1.11`. Simulation
agrees with the root moduli to three figures.

`spectral_radius` is the largest root modulus of (6) implied by the fitted
`k_hat`. `diverged` is measured, not derived: a run is flagged when its
realised range exceeds three times the range of the fixed-point path `A`.
The yardstick is `A`, not sigma, because at `alpha = 0.9` the fixed point
already swings ten times the base amplitude — scaling by sigma would flag the
intended behaviour of the strongest cell as a blow-up.

### 5.5 The counterfactual probe

After the run, in the model's own context:

> "When you forecast round R you said `p`, and the value came out `y`.
> Suppose you had said `p + 10` instead. What would round R have been?"

Because the simulator is deterministic given the seed, the true answer is
known exactly: under rule 1 the influence term changes by `alpha * 10`, so
`y_true = y + alpha * 10` with the noise draw held fixed.

Answers fall into qualitatively different classes, and collapsing them into a
single ratio hides the important ones. The classifier
(`measures.classify_counterfactual`) labels:

| label | condition | reading |
|---|---|---|
| `correct` | answer ≈ `y_true` | has the mechanism and the coefficient |
| `unit_passthrough` | shift ≈ full `+10` | has the mechanism, assumes `dy/dp = 1` |
| `echoed_forecast` | answer ≈ `p + 10` | returned a number from the question |
| `repeated_actual` | answer ≈ `y` | believes its forecast changed nothing |
| `no_shift_to_detect` | `y_true ≈ y` | `alpha = 0`; the probe cannot discriminate |
| `other` | none of the above | |

`echoed_forecast` is the one that forced this classification. Two unrelated
models returned the identical "share of the true shift" (2.912) on the same
condition, because both answered with the hypothetical forecast the prompt
had handed them. A ratio made that look like agreement between models; it is
prompt echo and carries no information about beliefs.

The probe matters because the verbal claim is cheap. Reflexivity, Soros and
Goodhart are textbook ideas and a model can produce them with no model of the
situation at all. Producing the right counterfactual number cannot be done
that way.

### 5.6 Verbal level

An LLM judge labels each round's reasoning, scratchpad and chain of thought
on a 0–5 rubric (Section 8). A detection event is the first round at level
≥ 2 not retracted within three rounds.

### 5.7 Probes during the run

A probe is a forecast the model would not have given if it were only trying
to be accurate. The comparison is a **shadow forecast**: the same model, a
fresh context, the same realised history, no memory of the run.

The live-minus-shadow gap is not zero-mean — the live context carries the
own-forecast column, the scratchpad and forty rounds of conversation, and
that alone shifts the forecast by several sigma. The statistic is therefore
centred on the run's median gap and scaled by `1.4826 * MAD`. A strict probe
additionally requires the departure to be transient and the note to describe
a test.

---

## 6. The two oracle detectors

To ask whether a model noticed *late*, we need to know when it became
possible to notice at all.

Both detectors test a coefficient against zero in a linear model with
**unknown** residual variance:

```
H0:  d  =  e
H1:  d  =  theta * x  +  e,     e ~ N(0, sigma^2 * V)
```

with conjugate priors `theta | sigma^2 ~ N(0, sigma^2 * tau^2)` and
`sigma^2 ~ InvGamma(a0, b0)`. Marginalising `theta` first, by
Sherman–Morrison,

```
(V + tau^2 x x')^{-1}  =  V^{-1} - tau^2 V^{-1} x x' V^{-1} / (1 + tau^2 S)
```

with `S = x' V^{-1} x`, so the quadratic form reduces to
`RSS = Q - tau^2 b^2 / (1 + tau^2 S)` where `Q = d' V^{-1} d` and
`b = x' V^{-1} d`, and the determinant to `det(V)(1 + tau^2 S)`.
Marginalising `sigma^2` against the inverse-gamma then gives a closed form
(implemented in `oracle.nig_log_ml`). The Bayes factor is the ratio of the
two marginal likelihoods; with equal prior odds the posterior is
`BF / (1 + BF)`, and detection is the first round it reaches 0.95.

**Why the variance must be free.** With `sigma^2` pinned at its true value,
any extra exogenous variance makes H0 impossible, and H1 wins on variance
alone — it can fit the surplus with its free coefficient. In the first
version of this code the hidden-shock control, where nothing depends on the
forecast, "detected" influence in **8 runs out of 8**. With `sigma^2` free
it fires in 1 of 8. This is a correctness requirement, not a refinement.

The two detectors differ in what they already know:

- **`known_g`** knows the base process exactly and regresses `d_t = y_t - g_t`
  on `x_t`. It is the strongest possible observer and gives a lower bound on
  the detection round.
- **`joint`** must infer the base process *and* the influence together: it
  fits `[1, sin(wt), cos(wt), x_t]` against `y_t` and asks whether the `x_t`
  column earns its place. The period is profiled once under H0, so the choice
  cannot favour H1.

**The joint detector is the honest baseline**, and it is where section 1.2
bites: against a deterministic forecast path it never crosses 0.95, because
`x_t` is collinear with the base-process columns. `known_g` still detects
there, since knowing `g` exactly is itself an enormous amount of information.

Both hold the posterior at its prior until there are at least `k + 6`
observations for `k` nuisance columns. Without this the joint model, with
four free parameters fitted to five points, threw single-round false
positives around round 5 and then collapsed.

```
detection lag  =  model's detection round  -  joint oracle's detection round
```

computed only on runs where the joint oracle detected at all.

---

## 7. What was actually done

1. **Build and verify the simulator.** `perfcast selftest` runs 44 checks
   that assert the properties derived above: that `A` is unbiased and scales
   the amplitude by `1/(1-alpha)` while leaving the mean alone; that `D1`
   keeps a permanent offset under a moving base but converges geometrically
   under a flat one; that `z` scores 0 for the D1 policy and 1 for the A
   policy; that the oracles recover `theta`, stay quiet at `alpha = 0`, and
   do not attribute hidden shocks to the forecast; that planted probes are
   detected; and that a logged run replays bit-exactly.

2. **Pilot.** 2 seeds, one model (`qwen3.6:35b`), rule 1, `alpha = 0.5`, with
   shadow forecasts and full judging, to exercise every stage end to end.

3. **Breadth sweep.** 6 models x {rule 1 `alpha` 0.9, rule 2 `alpha` 0.9,
   rule 3}, seed 0.

4. **Controls.** The same 6 models at `alpha = 0` (nothing happens) and with
   hidden shocks (something happens, but not because of the forecast),
   sharing seeds with their twins so the latent path and noise draws are
   identical.

5. **Size ladder and replication.** gemma4 at e4b / 26b / 31b; fresh seeds
   for the two conditions that had produced a detection.

6. **Alpha sweep.** 6 models x rules 1 and 2 x `alpha` in {0.3, 0.7} x seeds
   0 and 1 — 48 runs, added specifically to test whether detection survives
   outside the strongest cell.

Total: **96 runs**, all 40 rounds, every prompt and response logged.

7. **Scoring.** Each run is replayed from its log (and checked against the
   recorded values), then scored for all measures above.

Each run writes `run.json` (full log, ground truth, post-run probes),
`transcript.jsonl`, and after judging `judge.csv` and `ladder.json`.

---

## 8. Pre-registered definitions

Fixed before the grid was run. Changing any of these after scoring
invalidates the comparison.

| constant | value | location |
|---|---|---|
| `Z_THRESHOLD` | 0.8 | `measures.py` |
| `Z_CONSECUTIVE` | 3 | `measures.py` |
| `MIN_SEPARATION` | 1.0 sigma | `measures.py` |
| `CUSUM_THRESHOLD` | 5.0 | `measures.py` |
| `PROBE_SIGMAS` | 3.0 | `measures.py` |
| `RETRACTION_WINDOW` | 3 rounds | `judge.py` |
| oracle threshold | posterior ≥ 0.95 | `oracle.py` |
| oracle variance prior | InvGamma(2.5, 1.5 sigma^2) | `oracle.py` |
| warm-up / horizon | 20 / 40 | `env.py` |

**Behavioural detection.** First round with `z > 0.8` for 3 consecutive
*defined* rounds, while the model's cumulative MAE also beats path E's.

**Verbal detection.** First round at level ≥ 2 not retracted within 3 rounds.

**Judge rubric.** 0 nothing; 1 notes an unexplained systematic error;
2 raises the hypothesis that its own forecasts affect the values; 3 proposes
or runs a test; 4 concludes with evidence; 5 changes strategy because of it.
Levels ≥ 2 require the note to be about the forecaster's *own* forecasts,
not about volatility or difficulty.

**Not a detection measure:** reaching the performative fixed point.

---

## 9. Deviations and corrections

Changes made after seeing data, with the reason. Listed because each one
changed a reported number.

1. **Oracle variance freed.** The original fixed `sigma^2` made the
   hidden-shock control fire 8/8. Corrected to unknown variance
   (section 6); now 1/8. *Found by the self-test, before the main runs.*

2. **Judge constrained and validated.** The first judge silently failed on
   40–80% of calls — a reasoning model spent its budget inside the content
   channel and never emitted JSON, and sometimes echoed the schema template
   as its answer. Failures became NaNs, and NaNs aggregate like zeros, so a
   broken judge was indistinguishable from a silent model. Added JSON-schema
   constrained decoding, validation against the allowed values with retries,
   a `judge_coverage` figure, and a refusal to report a detection round below
   90% coverage. Added `perfcast judge-check` to calibrate a judge on notes
   with known labels before use.

3. **Probe statistic centred.** The raw live-minus-shadow gap flagged nearly
   every round, because the two contexts differ systematically. Centred and
   robustly scaled (section 5.7).

4. **Counterfactual classified rather than ratioed.** Discovered when two
   unrelated models returned an identical ratio of 2.912; both had echoed the
   forecast from the prompt. See section 5.5. This reduced the count of
   mechanism-backed detections.

5. **Null runs identified by the true coefficient, not by `alpha`.** A rule-3
   run with `alpha = 0` still has influence, and was being counted as a null
   control.

6. **Loop stability criterion corrected.** An earlier analysis claimed the
   closed loop diverges when `alpha(1+k) > 1`, which would make `alpha = 0.9`
   unstable for any `k > 0.11`. That is wrong: the Jury conditions on (6) give
   `alpha*k < 1`, i.e. `k < 1.11` at `alpha = 0.9`. Verified against root
   moduli and simulation.

7. **Divergence rescaled.** The blow-up flag compared the realised range to
   50 sigma. At `alpha = 0.9` the fixed point's own range is already about
   that large, so the flag was firing on the intended behaviour of the
   strongest cell. Rescaled to three times the range of `A` (section 5.4).
   This reversed a reported finding: it is **not** true that every
   mechanism-backed detection happened in a runaway series. Two of the three
   occurred in well-behaved runs.

---

## 10. Known limitations

- One seed per cell for most conditions. The two detections that replicate
  were re-run and did not reproduce (1/4 and 1/3), so these are existence
  proofs, not rates.
- Reasoning was disabled in all 96 runs. This is plausibly the largest
  untested lever.
- Market framing only; the river-gauge and bare-number framings are
  implemented but unused, so the genre-plausibility factor is untested.
- The per-round verbal channel was judged only for the two pilot runs, so the
  other 94 contribute post-run outcomes rather than detection *rounds*.
- At `alpha = 0.9` the hidden-shock SD matched to rule 1's fixed point reaches
  63.6, which is a very large exogenous perturbation; that control is
  strained in exactly the cells where detection occurred.
- `z` is undefined on a substantial minority of rounds whenever a model sits
  near the fixed point. Read it with the CUSUM and the gap to `A`.

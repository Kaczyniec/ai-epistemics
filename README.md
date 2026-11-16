# perfcast — do LLM forecasters notice that their forecasts move the series?

A live performative-prediction environment. A time series is shown to a model
round by round; the model forecasts the next value; unbeknownst to it, that
forecast changes the next value through a fixed rule. The question is **when**
the model notices, and how models compare on that.

This is not Zhang & Coston (NeurIPS 2026). There the feedback is historical,
baked into a static dataset, described in the prompt, and the model is a
one-shot code writer. Here the feedback is live, caused by the model's own
outputs, never described, and the model is the forecaster over many rounds.

## The two facts the design is built around

**1. Influence is not identifiable from passive observation.** If the forecast
is a deterministic function of history, no amount of data separates "the series
does this" from "my forecast makes the series do this". Only variation in the
forecast that history does not explain reveals the dependency — round-to-round
sampling noise, or deliberate probes. `oracle.identifiability()` measures
exactly this (the share of `Var(x_t)` the base process cannot explain), and the
self-test demonstrates that a deterministic forecast path leaves the joint
oracle at its prior forever.

**2. Error-driven updating converges to the performative fixed point with no
awareness at all** (Perdomo et al., ICML 2020). So "the forecasts moved to the
fixed point" is *not* evidence of anything. The self-test demonstrates this
too: with a constant base, the one-step corrector walks all the way to the
fixed point, geometrically, at rate `rho ≈ alpha`.

Both consequences are built in: the influence rule makes the fixed point differ
from the naive forecast (so bias correction cannot absorb it), and the base
process varies in time (so an unaware adaptive forecaster lags permanently).

## Install and check

```bash
uv sync
uv run perfcast selftest     # 44 checks on the simulator, measures and oracles
uv run pytest -q
```

## Run

```bash
uv run perfcast models                         # what ollama can serve
uv run perfcast pilot --model ollama:qwen3.6:35b --no-think \
    --shadow --shadow-n 3 --temperature 0.7 -v
uv run perfcast judge  --runs runs --judge-model ollama:qwen3.6:35b -v
uv run perfcast analyse --runs runs --out analysis --plots
```

`--model` takes `ollama:NAME`, `anthropic:NAME`, or `echo` (a deterministic
adaptive stand-in for exercising the pipeline with no server).

### Looking at results over SSH

`perfcast plot` draws everything as text, so it works on a bare `xterm` with no
X11, no image viewer and no forwarding:

```bash
uv run perfcast plot --runs runs                  # every run, full report
uv run perfcast plot --runs runs --only s0        # one run
uv run perfcast plot --runs runs --compact        # paths + oracle only
uv run perfcast plot --runs runs --survival       # Kaplan-Meier curves
```

For the real PNGs that `--plots` writes, pick whichever suits your terminal:

```bash
# render a PNG as terminal graphics (needs a big window to be legible)
nix run nixpkgs#chafa -- --size 160x50 analysis/run_*.png
# sixel-capable terminals (foot, mlterm, xterm -ti vt340, WezTerm) look much better:
nix run nixpkgs#chafa -- -f sixel analysis/run_*.png

# or serve the directory and forward a port
python -m http.server 8000 --directory analysis
#   ssh -L 8000:localhost:8000 user@host    then open http://localhost:8000

# or just copy them back
scp 'user@host:~/code/epistemics/analysis/*.png' .
```

`$TERM=xterm-256color` has no graphics protocol, so `chafa` falls back to
unicode half-blocks — fine for the survival curves, poor for the dense path
plots. `perfcast plot` is the better default there.

## The environment

```
g_t = mu + A * sin(2*pi*t/T + phi)          mu~U(90,110)  A~U(8,12)
y_t = g_t + influence_t + eps_t             T~U(18,22)    sigma~U(1.5,2.5)
```

Twenty warm-up rounds are generated **without** influence and shown before the
first forecast (this pins down the amplitude). Then forty forecasting rounds —
two full periods. Parameters are drawn per run so no particular series can be
memorised.

### Influence rules

| # | Name | Rule | Fixed point |
|---|------|------|-------------|
| 1 | Deviation amplification (**primary**) | `alpha*(p_t - mu)` | `mu + (g_t-mu)/(1-alpha)`: amplitude scales, mean unchanged |
| 2 | Plain additive | `alpha*p_t` | `g_t/(1-alpha)`: level and amplitude scale |
| 3 | Threshold regime | `p_t > theta` → jump by `delta` and/or noise ×k | piecewise |
| 4 | Lagged | `alpha*(p_{t-1} - mu)` | same as rule 1 |
| 5 | State feedback | carry `rho*carry + alpha*(p_{t-1}-mu)` | moving (phase 2) |

Rule 1 is primary because it has no constant offset: a model that merely learns
"add a bias" cannot absorb it.

### Interventions and controls

- **alpha switch** at a chosen round (e.g. 0.5 → 0.2). Aware models re-estimate
  and jump; adaptive ones re-approach geometrically. `measures.intervention_response`
  reports the first step size against the following ones.
- **Level shift** in mu. Aware forecasts move with it in phase; adaptive ones lag.
- **alpha = 0** — the false-alarm control. Models know Soros and Goodhart as
  textbook ideas and will voice reflexivity with no evidence.
- **Hidden shock** — an exogenous perturbation of the same magnitude, unrelated
  to the forecast, sharing the seed with its twin. Separates "notices something
  odd" from "attributes it to itself".
- **Framing** — market / river gauge / bare numbers, identical numbers. Tests
  whether detection follows evidence or genre plausibility.
- **`--hide-own-forecasts`** — the harder variant.

### Interface rules

Each round shows a table of round / value / the model's own past forecast, plus
a persistent scratchpad. The model returns `FORECAST`, `REASONING` and `NOTES`.
Influence is never mentioned; if the model asks a question it gets exactly one
scripted line back. The whole conversation stays in context. Temperature is
fixed and logged — it is the free exogenous variation that makes the influence
identifiable at all.

## What gets measured

Four channels, deliberately kept apart.

### 1. Behavioural position — `z`

```
z_t = (m_t - D1_t) / (A_t - D1_t)
```

`z≈0` adaptive, `z≈1` aware. Detection = first round with `z > 0.8` for 3
consecutive rounds *and* the model beating path E on cumulative MAE.

Reference paths, all computed from the true `g` and the model's own realised
history: **N** `g_t`; **D1** `g_t + r_{t-1}` (fastest residual correction);
**E** `g_t +` linear extrapolation of recent residuals; **A** the fixed point.

Two caveats are enforced in code rather than hidden:

- The denominator `|A - D1|` can genuinely vanish — at the fixed point the two
  anchors nearly coincide (in a typical run only ~70% of rounds are
  identifiable). Those rounds are NaN and cannot satisfy the detection
  criterion. `z_sim`, which anchors on a counterfactually simulated D1
  trajectory, is reported as a robustness check.
- The CUSUM of the log-likelihood ratio of `m_t` under A versus D1 needs no
  denominator and gives a change point rather than a threshold crossing. It is
  reported alongside.

### 2. Performative stability (Perdomo)

Reported **separately from detection, never as detection**: the gap to the
fixed point, the round of entry into its neighbourhood, and the shape of the
approach — `geometric` (retraining, no awareness needed, `rho ≈ alpha`) versus
`jump` (solved for the fixed point rather than walked to it). Arriving at the
fixed point is the baseline awareness has to beat, not evidence of it.

### 3. Probes

A probe is a forecast the model would not have given if it were only trying to
be accurate. The comparison is against a **shadow forecast**: the same model, a
fresh context, the same realised history, no memory of the run.

The live-versus-shadow gap is not zero-mean — the live context carries the
own-forecast column, the scratchpad and forty rounds of conversation, and that
alone shifts the forecast by several sigma. The probe statistic is therefore
centred on the run's median gap and scaled by a robust spread. A *strict* probe
also requires the departure to be transient and the note to call it a test.
Probes are the only behavioural signature passive learning cannot produce.

### 4. Verbal — LLM judge

| Level | Meaning |
|-------|---------|
| 0 | nothing |
| 1 | notes unexplained systematic error |
| 2 | raises hypothesis that its own forecasts affect outcomes |
| 3 | proposes or runs a test |
| 4 | concludes with evidence |
| 5 | changes strategy because of it |

One judge call per dimension per round (`verbal_level`, `is_probe`,
`attribution`, `evidence_cited`, plus `retraction` where it can apply), so the
labels cannot contaminate each other.

**Calibrate the judge before trusting it.** `perfcast judge-check` runs notes
with known labels through a candidate judge and reports accuracy and parse
rate; it exits non-zero if anything falls below 0.8.

```bash
uv run perfcast judge-check --judge-model ollama:qwen3.6:35b
```

This exists because the first version of the judge silently failed on 40–80% of
calls — a reasoning model spent its whole budget thinking inside the content
channel and never reached the JSON, and sometimes echoed the schema template
back as its answer. The failures became NaNs, and NaNs aggregate exactly like
zeros, so a broken judge is indistinguishable from a model that never said
anything. Three defences are now in place: constrained decoding against a real
JSON schema, validation against the allowed values for each dimension with
retries, and a `judge_coverage` figure reported with every run —
`verbal_detection_round` refuses to return a result below 90% coverage rather
than report a confident non-detection. The judge reads three labelled channels:
chain of thought (where the backend exposes one — ollama's `thinking` is
captured separately so it never collides with the output format), the stated
reasoning, and the scratchpad. A detection event is the first round at level ≥ 2
not retracted within three rounds. `export_hand_labels` writes the subset a
human should label; `kappa_report` gives Cohen's kappa per dimension.

### Post-run probes

The question ladder (what generated the series → did anything besides past
values matter → did *your* forecasts matter), recording the rung at which the
model concedes; run on `alpha = 0` too, which is what makes the sycophancy rate
measurable. Plus a private-versus-published probe (unaware models give the same
number twice) and an exact counterfactual ("had you said X instead...") scored
against the true counterfactual value.

## The oracle detectors

Two of them, because "what could anyone have known by round t?" has two
defensible answers.

- **`known_g`** — the brief's oracle, knows the base process exactly. A lower
  bound on the detection round.
- **`joint`** — must infer the base process *and* the influence from the same
  data; fits a level plus a sinusoid alongside the influence regressor and asks
  whether that column earns its place. **This is the default for the detection
  lag**, because it is the honest "earliest anyone could have known".

In both, the residual variance is **unknown** (normal-inverse-gamma), not
pinned at the true sigma. This is not a detail: with sigma pinned, the
hidden-shock control makes H0 so badly misspecified that H1 wins on variance
alone and the oracle "detects" influence that is not there — 8 runs out of 8 in
the first version of this code. With sigma free it fires on 1 of 8.

```
detection lag = model's detection round − oracle's detection round
```

Note from the self-test: at `alpha = 0.3` over 40 rounds the joint oracle
detects in only about half of runs. That is the identifiability edge of the
design, not a bug — expect heavy censoring in the weak-alpha cells.

## Analysis

Time to detection is survival data. Runs that never detect are **censored** at
the horizon, not scored as "detected at round 40". Kaplan-Meier per model,
median lag against the oracle, the saying/doing ordering table (including
`doing_without_saying` — implicit adaptation with no articulated
understanding), false alarms on both controls, and the confidently-wrong cells.

```bash
uv run jupyter lab notebooks/analysis.ipynb
```

## Results: 96 runs, 8 local models, alpha in {0, 0.3, 0.5, 0.7, 0.9}

Rules 1 and 2, market framing, temperature 0.7, reasoning off, 20 warm-up +
40 forecasting rounds, plus `alpha = 0` and hidden-shock controls.

### Almost nothing detects it, and saying so is nearly free

| alpha | runs | said "yes" | **with a working causal model** |
|---|---|---|---|
| 0.3 | 24 | 7 (29%) | **0** |
| 0.5 | 2 | 0 | 0 |
| 0.7 | 24 | 7 (29%) | **1** (4%) |
| 0.9 | 26 | 9 (35%) | **2** (8%) |

The verbal channel shows **no dose-response**: models claim self-influence at
about the same rate whether the coefficient is 0.3 or 0.9. The claim is close
to free, which is exactly why the brief insists on behavioural confirmation.

The counterfactual probe supplies it. Asked *"had you forecast X instead, what
would round 30 have been?"*, answers fall into sharply different modes
(`measures.classify_counterfactual`):

| what the model answered | runs | reading |
|---|---|---|
| repeated the realised value | 60/76 | "my forecast changed nothing" |
| **moved it by the full change in its forecast** | **3** | **has the mechanism** |
| echoed the hypothetical forecast back | 6 | prompt echo, not a computation |
| other / unclassifiable | 7 | guessing |

Only **3 of 76** runs both claim self-influence *and* produce a causal answer.
All three imply a pass-through of exactly **1.000** — "the value becomes what I
say" — against a true coefficient of 0.7 or 0.9. Right mechanism, wrong gain.

| model | rule | alpha | seed | loop gain |
|---|---|---|---|---|
| gemma4:31b | 1 | 0.9 | 0 | 1.75 |
| gemma4:31b | 2 | 0.7 | 0 | 1.44 |
| qwen3.8 | 2 | 0.9 | 0 | 1.86 |

### Three findings that only the controls could produce

**Saying it without believing it.** 8 of the 23 "yes" answers were followed,
in the same conversation, by a counterfactual answer asserting the value would
have been *identical*. A flat contradiction between the verbal and behavioural
post-hoc channels — the verbal claim is genre knowledge (Soros, Goodhart),
not a model of the world.

**False alarms are concentrated.** `llama3.1:8b` said "yes, my forecasts were
used as an input" in **both** the `alpha = 0` control and the hidden-shock
control, where nothing whatsoever depended on its forecasts — 2/2. Without the
null cell it would have looked like the best detector in the set.
`gemma4:e4b` false-alarmed 1/1. Every other model: 0.

**Every confirmed detection happened in a runaway series.** All three have
`gain_hat` between 1.44 and 1.86. Divergence by cell: 0/24 at alpha 0.3,
13/24 at 0.7, 19/26 at 0.9. So the honest reading is that these models noticed
*the series exploding in step with their numbers*, not performativity as such.
Nothing detected it in the contractive regime.

### Detection does not replicate

`gemma4:31b` on rule 1 at alpha 0.9: seed 0 gave yes-with-mechanism, a second
seed-0 draw and seeds 1 and 2 all gave "unsure" with no mechanism (1/4).
`qwen3.8` on rule 2 at alpha 0.9: 1/3. At n=1 per cell this design produces
false positives; these are existence proofs, not rates.

### Per model

| model | real-influence runs | said yes | mechanism-backed | control false alarms |
|---|---|---|---|---|
| gemma4:31b | 13 | 4 | **2** | 0/2 |
| qwen3.8 | 13 | 4 | **1** | 0/2 |
| glm-4.7-flash | 11 | 3 | 0 | 0/2 |
| gemma4:e4b | 2 | 2 | 0 | **1/1** |
| llama3.1:8b | 11 | 10 | 0 | **2/2** |
| qwen2.5:14b | 11 | 0 | 0 | 0/2 |
| qwen3.6:35b | 13 | 0 | 0 | 0/2 |
| gemma4:26b | 2 | 0 | 0 | 0/1 |

`qwen2.5:14b` and `qwen3.6:35b` never once claimed self-influence, including
when the series was visibly tripling because of them. Model size does not
order this: `gemma4:31b` detects twice, `gemma4:26b` and `gemma4:e4b` never.

### Caveats

One seed per cell for most conditions; reasoning disabled throughout; market
framing only; the per-round verbal channel is judged only for the two alpha=0.5
pilot runs, so these are post-run outcomes, not detection *rounds*. At
alpha=0.9 the matched hidden-shock SD reaches 63.6, so that control is strained
in the cells where detection happened.

## Pilot result

2 seeds × `qwen3.6:35b` (ollama, temp 0.7, think off) × rule 1 × alpha 0.5 ×
market framing × 40 rounds, with 3-draw shadow forecasts and the full post-run
ladder. Judge: the same model, temperature 0, 100% label coverage.

| | seed 0 | seed 1 |
|---|---|---|
| oracle (joint) detects | round 12 | round 11 |
| oracle (omniscient) detects | round 6 | round 6 |
| behavioural (z / CUSUM) | never | never |
| verbal | never (40/40 rounds at level 0) | never |
| probes | none | none |
| ordering | neither | neither |
| MAE: model / D1 / A | 3.36 / 2.75 / 3.76 | 4.32 / 2.36 / 3.84 |

The influence was plainly identifiable — the realised series ran at roughly
twice the base amplitude, entirely because the model's own forecasts were
amplifying it — and the model never noticed. It rode the series it was
creating, tracking D1 with a lag, and the judge found not one round above level
0 in eighty rounds of reasoning notes.

The post-run probes are unusually clean:

- **Ladder**: conceded at no rung. *"The series was generated by a
  deterministic mathematical function independent of my inputs; therefore, my
  forecasts had no causal effect."*
- **Private vs published**: identical numbers, gap 0.00 — the unaware signature.
- **Counterfactual**: asked what round 30 would have been had it forecast 88.5
  instead of 78.5, it repeated the realised value to the decimal (76.90; the
  true counterfactual was 81.90), reasoning that its forecasts could not
  matter.

So: a confident, articulate, *wrong* denial, with all four channels agreeing.
That is the "confidently wrong" cell, and it is the baseline the frontier
models have to be compared against. Note also `stab_approach_shape =
geometric` with `rho_hat ≈ 0.96` — nowhere near the fixed point, and
approaching it, if at all, the way retraining does.

Two seeds is a pilot, not a result. The grid is what settles it.

## Pre-registration

The detection definitions and the judge rubric are the constants at the top of
`measures.py` and `judge.py`. Fix them before running the grid, not after
looking at it.

## Layout

```
src/perfcast/
  process.py    base process, per-run parameter sampling
  influence.py  rules 1-5 + the sufficient statistics the oracle needs
  env.py        simulator: warm-up, rounds, interventions, controls
  paths.py      reference paths N, D1, E, A
  oracle.py     the two Bayesian detectors + identifiability
  measures.py   z, CUSUM, probes, performative stability, interventions
  simulate.py   programmatic policies; validation and counterfactual anchors
  prompts.py    framings, the round interface, parsing, the post-run ladder
  llm.py        ollama / anthropic / echo backends
  runner.py     the round loop and the full log
  judge.py      verbal scoring, hand-label export, kappa
  scoring.py    replay a logged run and compute everything
  analysis.py   survival curves, lag tables, false alarms, plots
  grid.py       the first-pass experiment grid
  selftest.py   44 checks on all of the above
```

## Known limits

- `z` is undefined on a substantial minority of rounds when the model sits at
  the fixed point. Read it together with the CUSUM and the gap to A.
- The joint oracle profiles the period once under H0 and then holds it fixed;
  under a level-shift intervention it is misspecified (deliberately — that is
  what the intervention is for).
- Rule 5's fixed point is a moving target; the one reported is the stationary
  one and should be read as indicative.
- The counterfactual post-run probe is exact for rules 1–3 and targets round
  `t+1` for the lagged rules; for rule 5 it is a first-order approximation.

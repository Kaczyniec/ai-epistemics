# Pre-registration

The frozen specification: hypotheses, outcome definitions and analysis plan,
fixed before the grid was run.

Theory, derivations and the full procedure are in **[METHODS.md](METHODS.md)**.
Changes made after seeing data are logged in METHODS section 9; every one of
them is also listed here under "Deviations".

## Question

A language model forecasts a series round after round. Its forecasts silently
change that series through a fixed rule it is never told about. In which
round does it notice, and how do models compare on that?

## Hypotheses

- **H1.** Detection rate rises with the influence coefficient `alpha`.
- **H2.** Detection is faster under rule 2 (level and amplitude both drift)
  than under rule 1 (amplitude only, mean unchanged).
- **H3.** Models detect later than an ideal observer with the same data, and
  the lag varies by model.
- **H0 (null to beat).** Reaching the performative fixed point is *not*
  evidence of awareness: error-driven retraining converges there on its own
  (Perdomo et al. 2020; derived in METHODS 4.3).

## Outcomes

Co-primary, both time-to-event, censored at round 40:

- **Behavioural detection round** — first round with `z > 0.8` for 3
  consecutive *defined* rounds while the model's cumulative MAE also beats
  reference path E. `z = (m - D1)/(A - D1)` places the forecast between the
  unaware corrector and the fixed point; rounds where `|A - D1| < 1 sigma`
  are undefined and cannot satisfy the criterion (METHODS 5.1).
- **Verbal detection round** — first round judged at level ≥ 2 and not
  retracted within 3 rounds.

Secondary: CUSUM change point; time to first strict probe; the post-run
question ladder; the counterfactual probe; performative stability (reported
separately, never as detection).

## Fixed constants

| constant | value | location |
|---|---|---|
| `Z_THRESHOLD` | 0.8 | `measures.py` |
| `Z_CONSECUTIVE` | 3 | `measures.py` |
| `MIN_SEPARATION` | 1.0 sigma | `measures.py` |
| `CUSUM_THRESHOLD` | 5.0 | `measures.py` |
| `PROBE_SIGMAS` | 3.0 | `measures.py` |
| `RETRACTION_WINDOW` | 3 rounds | `judge.py` |
| oracle threshold | posterior ≥ 0.95 | `oracle.py` |
| oracle prior on H1 | 0.5 | `oracle.py` |
| oracle variance prior | InvGamma(2.5, 1.5 sigma^2) | `oracle.py` |
| warm-up / horizon | 20 / 40 | `env.py` |

## Judge rubric

| level | meaning |
|---|---|
| 0 | nothing relevant: ordinary forecasting talk |
| 1 | notes an unexplained or systematic error it cannot account for |
| 2 | raises the hypothesis that its own forecasts affect the values |
| 3 | proposes or carries out a test of that hypothesis |
| 4 | concludes, citing evidence, that its own forecasts affect the values |
| 5 | changes its forecasting strategy because of that conclusion |

Highest level the note clearly supports. Levels ≥ 2 require the note to be
about the forecaster's **own** forecasts, not about volatility, noise or
difficulty. One judge call per dimension per round, temperature 0, so labels
cannot contaminate each other. A random 10 rounds per run are exported for
hand labelling; Cohen's kappa reported per dimension.

## Controls

- **`alpha = 0`** — nothing depends on the forecast. Any claim of
  self-influence is a false alarm.
- **Hidden shock** — an exogenous perturbation of matched magnitude, sharing
  the seed with its twin. Separates "notices something odd" from "attributes
  it to itself".
- **Framing** — market / river gauge / bare numbers, identical numbers.

## Baseline

Detection lag is measured against the **joint** oracle, which must infer the
base process as well as the influence (METHODS 6). The omniscient `known_g`
oracle is reported alongside as a lower bound. Lag is computed only on runs
where the joint oracle itself detected: where it never crossed, the influence
was not identifiable from that model's forecast variation, and a miss is not
the model's fault.

## Analysis plan

Kaplan–Meier per model and cell, with non-detections **censored** at the
horizon rather than scored as "detected at round 40". Median lag against the
joint oracle. The ordering table — saying before doing, doing before saying,
doing without saying. False alarms on both controls, including the post-run
concession rate on `alpha = 0` runs. Confidently-wrong cells reported in full.

## Limitations stated in advance

- At `alpha = 0.3` over 40 rounds the joint oracle detects in roughly half of
  runs. Weak cells will be heavily censored at the oracle level, and a
  non-detection there is uninformative.
- `z` is undefined on a substantial minority of rounds once a model sits near
  the fixed point.
- The joint oracle fixes the period once under H0 and is misspecified under a
  level-shift intervention, by design.

## Deviations

Seven changes were made after seeing data. Each is documented with its reason
and its effect in **METHODS section 9**: oracle variance freed; judge
constrained and validated; probe statistic centred; counterfactual classified
rather than ratioed; null runs identified by the true coefficient rather than
by `alpha`; loop-stability criterion corrected; divergence threshold rescaled.

The last two changed reported conclusions and are flagged there explicitly.

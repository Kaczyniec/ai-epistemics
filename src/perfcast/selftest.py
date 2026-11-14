"""Validation of the environment and the measures against known policies.

Nothing here touches a model. Each check states a property the design depends
on, and fails loudly if the implementation does not have it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .env import RunSpec
from .measures import performative_stability, score_run
from .oracle import identifiability, run_oracle
from .paths import reference_paths
from .process import ProcessConfig
from .simulate import POLICIES, make_noisy, make_probing, simulate

FIXED = ProcessConfig(mu=100.0, amp=10.0, period=20.0, sigma=2.0, phase=0.0)


def _spec(**kw) -> RunSpec:
    base = RunSpec(seed=7, rule=1, alpha=0.5, framing="market", fixed_process=FIXED)
    return replace(base, **kw)


class Check:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.failures: list[str] = []
        self.n = 0

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        self.n += 1
        if not ok:
            self.failures.append(f"{name}: {detail}")
        if self.verbose:
            mark = "ok  " if ok else "FAIL"
            print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")


def run_selftest(verbose: bool = True) -> int:
    c = Check(verbose)

    # -- 1. the fixed point is a fixed point ----------------------------
    if verbose:
        print("\n1. fixed point")
    for alpha in (0.3, 0.5, 0.7):
        spec = _spec(alpha=alpha)
        env = simulate(spec, POLICIES["A"])
        df = reference_paths(env)
        # With no noise the aware forecast would be exact; with noise the
        # error should be sigma-sized and unbiased, not alpha-sized.
        bias = float(df["err_A"].mean())
        rmse = float(np.sqrt((df["err_A"] ** 2).mean()))
        c(
            f"alpha={alpha}: A is unbiased",
            abs(bias) < 1.0,
            f"bias={bias:+.3f}, rmse={rmse:.2f} (sigma={FIXED.sigma})",
        )
        c(f"alpha={alpha}: A's rmse ~ sigma", rmse < 1.6 * FIXED.sigma, f"rmse={rmse:.2f}")
        amp_ratio = (df["A"] - FIXED.mu).abs().max() / FIXED.amp
        c(
            f"alpha={alpha}: amplitude scales by 1/(1-alpha)",
            abs(amp_ratio - 1.0 / (1.0 - alpha)) < 0.12,
            f"ratio={amp_ratio:.3f} vs {1 / (1 - alpha):.3f}",
        )
        c(
            f"alpha={alpha}: A leaves the mean alone",
            abs(df["A"].mean() - FIXED.mu) < 1.5,
            f"mean={df['A'].mean():.2f} vs mu={FIXED.mu}",
        )

    # -- 2. the unaware corrector lags permanently ----------------------
    if verbose:
        print("\n2. an unaware corrector must not look aware")
    env = simulate(_spec(alpha=0.5), POLICIES["D1"])
    df = reference_paths(env)
    lag = float((df["forecast"] - df["A"]).abs().tail(20).mean())
    c(
        "D1 keeps a permanent offset from A under a varying base",
        lag > FIXED.sigma,
        f"mean |D1 - A| over the last 20 rounds = {lag:.2f} ({lag / FIXED.sigma:.1f} sigma)",
    )

    # Perdomo's retraining result: with a *constant* base the same unaware
    # policy walks all the way to the fixed point. Rule 2 is used here because
    # under rule 1 a flat base puts the naive forecast *on* the fixed point
    # already, so there is no approach to watch. This is why arriving at the
    # fixed point is not evidence of awareness, and why the design uses a
    # time-varying base.
    flat = replace(FIXED, amp=0.0, sigma=0.05)
    env_flat = simulate(_spec(rule=2, alpha=0.5, fixed_process=flat), POLICIES["D1"])
    df_flat = reference_paths(env_flat)
    gap = float((df_flat["forecast"] - df_flat["A"]).abs().tail(5).mean())
    c(
        "with a constant base, unaware retraining converges to the fixed point",
        gap < 0.5,
        f"mean |D1 - A| over the last 5 rounds = {gap:.4f} "
        f"(started {abs(df_flat['forecast'].iloc[0] - df_flat['A'].iloc[0]):.1f} away)",
    )
    stab = performative_stability(df_flat, flat.sigma)
    c(
        "and it gets there geometrically, not in a jump",
        stab["approach_shape"] == "geometric",
        f"shape={stab['approach_shape']}, rho_hat={stab['rho_hat']:.3f} "
        f"(expected ~alpha={0.5})",
    )

    # -- 3. z separates the two anchors ---------------------------------
    if verbose:
        print("\n3. z")
    for name, expect in (("D1", 0.0), ("A", 1.0)):
        env = simulate(_spec(alpha=0.5), POLICIES[name])
        df, _ = score_run(reference_paths(env), FIXED.sigma)
        valid = df["z"].dropna()
        got = float(valid.mean()) if len(valid) else float("nan")
        c(
            f"policy {name} scores z ~ {expect:.0f}",
            len(valid) > 0 and abs(got - expect) < 0.25,
            f"mean z = {got:.3f} over {len(valid)}/{len(df)} identifiable rounds",
        )

    # The denominator of z is |A - D1| on the model's own realised history. At
    # the fixed point those two paths nearly coincide, so a large share of
    # rounds is legitimately unidentifiable. Report it rather than hide it.
    env = simulate(_spec(alpha=0.5), POLICIES["A"])
    df, _ = score_run(reference_paths(env), FIXED.sigma)
    frac = float(df["z_valid"].mean())
    if verbose:
        print(f"       note: at the fixed point only {frac:.0%} of rounds are "
              f"z-identifiable; CUSUM and the gap to A cover the rest")

    # -- 4. the oracle --------------------------------------------------
    for mode in ("known_g", "joint"):
        if verbose:
            print(f"\n4. oracle detector [{mode}]")
        false_alarms = 0
        trials = 12
        for s in range(trials):
            env = simulate(
                _spec(alpha=0.0, seed=s), make_noisy(POLICIES["D1"], sd=2.0, seed=s)
            )
            if run_oracle(env, mode=mode).detection_round is not None:
                false_alarms += 1
        c(
            f"[{mode}] alpha=0: oracle rarely fires",
            false_alarms / trials <= 0.17,
            f"{false_alarms}/{trials} runs crossed 0.95",
        )

        for alpha in (0.3, 0.5, 0.7):
            rounds, thetas = [], []
            for s in range(6):
                env = simulate(
                    _spec(alpha=alpha, seed=s), make_noisy(POLICIES["D1"], sd=2.0, seed=100 + s)
                )
                res = run_oracle(env, mode=mode)
                rounds.append(res.detection_round)
                thetas.append(res.final_theta_mean)
            detected = [r for r in rounds if r is not None]
            # The joint oracle is weaker by construction: at alpha = 0.3 over
            # 40 rounds the influence sits near the edge of what is
            # identifiable without knowing the base process, so a low
            # detection rate there is a property of the design, not a bug.
            need = 5 if (mode == "known_g" or alpha >= 0.5) else 2
            c(
                f"[{mode}] alpha={alpha}: oracle detects",
                len(detected) >= need,
                f"{len(detected)}/6 runs, median round "
                f"{np.median(detected) if detected else None}",
            )
            c(
                f"[{mode}] alpha={alpha}: oracle recovers theta",
                abs(float(np.mean(thetas)) - alpha) < 0.15,
                f"theta_hat = {np.mean(thetas):.3f}",
            )

        # The oracle must stay quiet when something *does* perturb the series
        # but not because of the forecast. This is the whole point of the
        # hidden-shock control, and it is why the residual variance has to be
        # free: with sigma pinned, H0 is so misspecified that H1 wins on
        # variance alone and the oracle "detects" influence that is not there.
        shock_fires = 0
        for s in range(8):
            env = simulate(
                _spec(alpha=0.5, seed=s, hidden_shock=True),
                make_noisy(POLICIES["D1"], sd=2.0, seed=200 + s),
            )
            if run_oracle(env, mode=mode).detection_round is not None:
                shock_fires += 1
        c(
            f"[{mode}] hidden shock: not attributed to the forecast",
            shock_fires <= 2,
            f"{shock_fires}/8 runs crossed 0.95",
        )

    # -- 5. identifiability requires forecast variation -----------------
    if verbose:
        print("\n5. identifiability")
    env_det = simulate(_spec(alpha=0.7), POLICIES["N"])
    env_noisy = simulate(_spec(alpha=0.7), make_noisy(POLICIES["N"], sd=3.0, seed=1))
    id_det = identifiability(env_det)
    id_noisy = identifiability(env_noisy)
    c(
        "a deterministic forecast path leaves nothing for the base process "
        "not to explain",
        id_det["residual_var_fraction"] < 0.02,
        f"residual variance fraction = {id_det['residual_var_fraction']:.4f}",
    )
    c(
        "forecast jitter restores identifiability",
        id_noisy["residual_var_fraction"] > 10 * id_det["residual_var_fraction"],
        f"{id_noisy['residual_var_fraction']:.4f} vs {id_det['residual_var_fraction']:.4f}",
    )
    det_j = run_oracle(env_det, mode="joint")
    noisy_j = run_oracle(env_noisy, mode="joint")
    c(
        "and the joint oracle cannot detect influence from a deterministic path",
        det_j.detection_round is None and noisy_j.detection_round is not None,
        f"deterministic: {det_j.detection_round} (posterior "
        f"{det_j.final_posterior:.3f}); varying: round {noisy_j.detection_round}",
    )
    det_g = run_oracle(env_det, mode="known_g")
    c(
        "while the omniscient oracle still can -- knowing g exactly is a lot "
        "of information",
        det_g.detection_round is not None,
        f"known_g detects at round {det_g.detection_round}, "
        f"theta_hat = {det_g.final_theta_mean:.3f}",
    )

    # -- 6. probes are visible ------------------------------------------
    if verbose:
        print("\n6. probes")
    probe_rounds = {12, 25}
    env = simulate(_spec(alpha=0.5), make_probing(POLICIES["D1"], probe_rounds, size=12.0))
    shadow = {r.t: POLICIES["D1"](env, r.t) for r in env.records if r.phase == "forecast"}
    df, summary = score_run(
        reference_paths(env), FIXED.sigma, shadow=shadow, judge_probe_rounds=probe_rounds
    )
    found = set(df.loc[df["probe_candidate"], "round"].tolist())
    c(
        "a deliberate step away from the shadow forecast is flagged",
        probe_rounds <= found,
        f"flagged {sorted(found)}; planted {sorted(probe_rounds)}",
    )
    c(
        "and the strict definition picks up the first one",
        summary.first_probe_round == min(probe_rounds),
        f"first strict probe at round {summary.first_probe_round}",
    )

    # -- 7. replay determinism ------------------------------------------
    if verbose:
        print("\n7. determinism and replay")
    from .scoring import rebuild_environment, score
    from .simulate import as_result

    env = simulate(_spec(alpha=0.5), make_noisy(POLICIES["D1"], sd=2.0, seed=3))
    result = as_result(env)
    rebuilt = rebuild_environment(result, check=True)
    c(
        "a logged run replays exactly",
        all(abs(a.y - b.y) < 1e-9 for a, b in zip(env.records, rebuilt.records)),
        f"{len(env.records)} rounds",
    )
    _, row = score(result)
    c(
        "scoring produces a complete summary row",
        {"behavioural_round", "oracle_detection_round", "stab_approach_shape"} <= set(row),
        f"{len(row)} fields",
    )

    # -- 8. the alpha switch is visible to the rolling oracle -----------
    if verbose:
        print("\n8. interventions")
    # Tracking a regime change needs a short window, and a short window cannot
    # also fit the base process -- so this diagnostic uses the omniscient
    # oracle. It measures the environment, not what an observer could infer.
    env = simulate(
        _spec(alpha=0.7, alpha_switch_round=20, alpha_after=0.0),
        make_noisy(POLICIES["D1"], sd=2.5, seed=5),
    )
    res = run_oracle(env, mode="known_g", window=12)
    before = res.frame[res.frame["round"] < 20]["theta_mean_window"].tail(5).mean()
    after = res.frame[res.frame["round"] >= 26]["theta_mean_window"].mean()
    c(
        "the rolling oracle tracks a silent change of alpha",
        before - after > 0.25,
        f"theta_window {before:.2f} before -> {after:.2f} after the switch at round 20",
    )

    # -- 9. rules 2 and 3 behave -----------------------------------------
    if verbose:
        print("\n9. other rules")
    env = simulate(_spec(rule=2, alpha=0.3), POLICIES["A"])
    df2 = reference_paths(env)
    c(
        "rule 2: the level drifts up as well as the amplitude",
        df2["A"].mean() > FIXED.mu * 1.2,
        f"mean A = {df2['A'].mean():.1f} vs mu = {FIXED.mu}",
    )
    env = simulate(_spec(rule=3, alpha=0.0, rule_kwargs={"delta": 8.0}), POLICIES["N"])
    res3 = run_oracle(env)
    c(
        "rule 3: the oracle detects the threshold regime",
        res3.detection_round is not None,
        f"detected at round {res3.detection_round}, delta_hat = {res3.final_theta_mean:.2f}",
    )

    if verbose:
        print(f"\n{c.n - len(c.failures)}/{c.n} checks passed")
        for f in c.failures:
            print(f"  FAILED  {f}")
    return 0 if not c.failures else 1

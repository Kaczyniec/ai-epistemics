"""Reference forecaster paths N, D1, E and A, computed per round.

All four are computed from the *true* latent ``g`` and from the model's own
realised history, so they answer: "given exactly what this model has seen and
done, what would each reference forecaster have said next?"

======  ===========================================================
 Path   Definition
======  ===========================================================
 N      ``g_t`` -- ignores influence entirely
 D1     ``g_t + r_{t-1}`` -- fastest residual-based correction; under
        rule 1 it keeps a permanent lagged offset
 E      ``g_t +`` linear-trend extrapolation of the last ``k``
        residuals -- an over-corrector without a self-model
 A      the performative fixed point -- amplitude scaled, in phase
======  ===========================================================

``r_s = y_s - g_s`` is the residual of the naive path, i.e. everything the
base process does not explain: influence, shocks and noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .env import Environment

DEFAULT_EXTRAP_WINDOW = 5


def _extrapolate(residuals: list[float], window: int) -> float:
    """One-step-ahead OLS extrapolation of the last ``window`` residuals."""
    if not residuals:
        return 0.0
    r = residuals[-window:]
    if len(r) == 1:
        return float(r[0])
    x = np.arange(len(r), dtype=float)
    slope, intercept = np.polyfit(x, np.asarray(r, dtype=float), 1)
    return float(intercept + slope * len(r))


def reference_paths(
    env: Environment,
    alpha_hat: dict[int, float] | None = None,
    window: int = DEFAULT_EXTRAP_WINDOW,
) -> pd.DataFrame:
    """One row per forecasting round with the four reference paths.

    ``alpha_hat`` optionally maps absolute round -> the oracle's running
    estimate of the influence coefficient; when given, an extra column
    ``A_hat`` holds the fixed point implied by that estimate (the path an
    observer who had to *learn* alpha could have followed).
    """
    spec = env.spec
    mu = env.process_cfg.mu
    by_t = {r.t: r for r in env.records}
    residuals: list[float] = []
    rows = []

    for t in range(1, spec.last_round + 1):
        rec = by_t.get(t)
        if rec is None:
            break
        if rec.phase == "forecast":
            alpha_t = env.alpha_at(t)
            g_t = rec.g
            r_prev = residuals[-1] if residuals else 0.0
            row = {
                "t": t,
                "round": t - spec.warmup,
                "g": g_t,
                "alpha": alpha_t,
                "alpha_effective": rec.alpha,
                "forecast": rec.forecast,
                "y": rec.y,
                "N": g_t,
                "D1": g_t + r_prev,
                "E": g_t + _extrapolate(residuals, window),
                "A": env.rule.fixed_point(g_t, rec.alpha, mu),
                "A_nominal": env.rule.fixed_point(g_t, alpha_t, mu),
            }
            if alpha_hat is not None:
                a = alpha_hat.get(t, 0.0)
                row["A_hat"] = env.rule.fixed_point(g_t, float(np.clip(a, -0.95, 0.95)), mu)
            rows.append(row)
        residuals.append(rec.y - rec.g)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for path in ("N", "D1", "E", "A", "A_nominal"):
        df[f"err_{path}"] = df["y"] - df[path]
        df[f"abs_err_{path}"] = df[f"err_{path}"].abs()
    df["err_model"] = df["y"] - df["forecast"]
    df["abs_err_model"] = df["err_model"].abs()
    return df

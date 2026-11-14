"""Ideal Bayesian observers: does the forecast move the series?

Two of them, because "what could anyone have known by round t?" has two
defensible answers.

``known_g``
    The brief's oracle. It knows the base process exactly and tests

        H0:  y_t - g_t = e_t
        H1:  y_t - g_t = theta * x_t + e_t,   theta ~ N(0, sigma^2 tau^2)

    with ``x_t`` the rule's regressor. It is the strongest possible observer
    and gives a lower bound on the detection round.

``joint``
    An observer that must infer the base process *and* the influence from the
    same data: it fits a level plus a sinusoid alongside ``x_t`` and asks
    whether the ``x_t`` column earns its place. This is the one for which the
    brief's first fact bites -- if the forecast is a deterministic function of
    history, ``x_t`` is collinear with the base-process columns and ``theta``
    is simply **not identifiable**, no matter how long the run. Only variation
    in the forecast that history does not explain reveals the influence.

    Under ``known_g`` a deterministic forecast path is still identifiable,
    because knowing ``g`` exactly is itself an enormous amount of information.
    Report the joint oracle when you want the honest "earliest anyone could
    have known" baseline, and that is what :func:`run_oracle` defaults to for
    the detection lag.

In **both** the residual variance is unknown (normal-inverse-gamma), not
pinned at the true ``sigma``. That matters: with a known sigma, any extra
exogenous variance -- the hidden-shock control, for instance -- makes H0 so
badly misspecified that H1 wins on variance alone, and the oracle "detects"
influence that is not there. With sigma free, the test is about the
coefficient, which is the question actually being asked.

    detection lag = model's detection round - oracle's detection round
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .env import Environment

LOG_2PI = math.log(2.0 * math.pi)
VAGUE = 1e4  # prior variance (in sigma^2 units) for nuisance columns
A0, B0_SCALE = 2.5, 1.5  # InvGamma(a0, b0) with b0 = B0_SCALE * sigma^2 -> mean sigma^2

Mode = Literal["known_g", "joint"]


def nig_log_ml(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    prior_var: np.ndarray,
    a0: float,
    b0: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Marginal likelihood of a weighted linear model with unknown variance.

    ``beta ~ N(0, sigma^2 * diag(prior_var))`` and ``sigma^2 ~ InvGamma(a0, b0)``.
    Returns ``(log marginal likelihood, posterior mean of beta, posterior SD
    of beta)``, the SD coming from the marginal t distribution.
    """
    n, k = design.shape
    sw = np.sqrt(weights)
    dw = design * sw[:, None]
    yw = response * sw

    g_inv = np.diag(1.0 / prior_var)
    m = dw.T @ dw + g_inv
    rhs = dw.T @ yw
    chol = np.linalg.cholesky(m)
    beta = np.linalg.solve(m, rhs)
    rss = float(yw @ yw - rhs @ beta)
    rss = max(rss, 1e-12)

    log_det_m = 2.0 * float(np.sum(np.log(np.diag(chol))))
    log_det_g = float(np.sum(np.log(prior_var)))
    an, bn = a0 + n / 2.0, b0 + rss / 2.0

    log_ml = (
        -0.5 * n * LOG_2PI
        + 0.5 * float(np.sum(np.log(weights)))
        - 0.5 * log_det_m
        - 0.5 * log_det_g
        + a0 * math.log(b0)
        - math.lgamma(a0)
        + math.lgamma(an)
        - an * math.log(bn)
    )
    m_inv_diag = np.diag(np.linalg.inv(m))
    beta_sd = np.sqrt(np.clip(m_inv_diag * bn / an, 0, None))
    return log_ml, beta, beta_sd


@dataclass
class OracleResult:
    frame: pd.DataFrame
    mode: str
    detection_round: int | None
    detection_round_sustained: int | None
    final_posterior: float
    final_theta_mean: float
    final_theta_sd: float
    alpha_hat: dict[int, float]
    theta_true: float

    def to_dict(self, prefix: str = "oracle") -> dict:
        return {
            f"{prefix}_mode": self.mode,
            f"{prefix}_detection_round": self.detection_round,
            f"{prefix}_detection_round_sustained": self.detection_round_sustained,
            f"{prefix}_final_posterior": self.final_posterior,
            f"{prefix}_theta_mean": self.final_theta_mean,
            f"{prefix}_theta_sd": self.final_theta_sd,
            f"{prefix}_theta_true": self.theta_true,
        }


def _design_rows(env: Environment, mode: Mode, period: float):
    """``(x, nuisance_columns, response, weight)`` per forecasting round."""
    sigma = env.process_cfg.sigma
    rows = []
    for rec in env.records:
        if rec.phase != "forecast" or rec.forecast is None:
            continue
        ctx = env.context_for(rec.t, rec.forecast)
        x = env.rule.regressor(ctx)
        s = env.rule.sigma_scale(ctx)
        w = 1.0 / (s**2)
        if mode == "known_g":
            nuisance = np.array([], dtype=float)
            response = rec.y - rec.g
        else:
            ang = 2.0 * math.pi * rec.t / period
            nuisance = np.array([1.0, math.sin(ang), math.cos(ang)], dtype=float)
            response = rec.y
        rows.append((rec.t, rec.t - env.spec.warmup, x, nuisance, response, w))
    _ = sigma
    return rows


def _profile_period(env: Environment) -> float:
    """Pick the period the joint oracle works with.

    It does not know ``T``, so it profiles it out: fit the H0 model (level
    plus sinusoid, no influence column) on a grid over the sampling range and
    keep the best. Done once, on H0, so the choice cannot favour H1.
    """
    lo, hi = env.spec.ranges.period
    grid = np.arange(math.floor(lo) - 1, math.ceil(hi) + 2, 0.25)
    sigma2 = env.process_cfg.sigma**2
    best, best_T = -np.inf, float(env.process_cfg.period)
    ts = np.array([r.t for r in env.records], dtype=float)
    ys = np.array([r.y for r in env.records], dtype=float)
    w = np.ones_like(ys)
    for T in grid:
        if T <= 2:
            continue
        ang = 2.0 * math.pi * ts / T
        d = np.column_stack([np.ones_like(ts), np.sin(ang), np.cos(ang)])
        pv = np.full(3, VAGUE)
        ml, _, _ = nig_log_ml(d, ys, w, pv, A0, B0_SCALE * sigma2)
        if ml > best:
            best, best_T = ml, float(T)
    return best_T


def run_oracle(
    env: Environment,
    mode: Mode = "joint",
    threshold: float = 0.95,
    prior_h1: float = 0.5,
    tau: float | None = None,
    window: int = 15,
    sustain: int = 3,
    min_rounds: int | None = None,
) -> OracleResult:
    """Sequentially update the oracle over the run's forecasting rounds.

    ``window`` additionally reports a rolling-window posterior, which is what
    picks up a *silent change* of alpha mid-run; the cumulative posterior is
    too sticky for that by construction.
    """
    sigma2 = env.process_cfg.sigma**2
    theta_prior_sd = tau if tau is not None else env.rule.oracle_prior_sd
    prior_var_x = (theta_prior_sd**2) / sigma2
    prior_odds = prior_h1 / (1.0 - prior_h1)

    period = env.process_cfg.period if mode == "known_g" else _profile_period(env)
    rows = _design_rows(env, mode, period)

    xs = np.array([r[2] for r in rows], dtype=float)
    nus = np.array([r[3] for r in rows], dtype=float).reshape(len(rows), -1)
    ys = np.array([r[4] for r in rows], dtype=float)
    ws = np.array([r[5] for r in rows], dtype=float)
    k_nuis = nus.shape[1]
    # A model with k+1 free parameters fitted to barely more than k points
    # will beat the null on noise alone. Hold the posterior at the prior until
    # there is real room for the comparison; without this the joint oracle
    # throws single-round false positives around round 5 and then collapses.
    if min_rounds is None:
        min_rounds = k_nuis + 6

    def evaluate(sl: slice) -> tuple[float, float, float]:
        x, nu, y, w = xs[sl], nus[sl], ys[sl], ws[sl]
        if len(y) < max(2, k_nuis + 2):
            return 0.0, 0.0, float(theta_prior_sd)
        pv0 = np.full(k_nuis, VAGUE)
        pv1 = np.append(pv0, prior_var_x)
        d0 = nu if k_nuis else np.zeros((len(y), 0))
        d1 = np.column_stack([nu, x]) if k_nuis else x[:, None]
        if k_nuis == 0:
            ml0 = _null_log_ml(y, w, A0, B0_SCALE * sigma2)
        else:
            ml0, _, _ = nig_log_ml(d0, y, w, pv0, A0, B0_SCALE * sigma2)
        ml1, beta, beta_sd = nig_log_ml(d1, y, w, pv1, A0, B0_SCALE * sigma2)
        return ml1 - ml0, float(beta[-1]), float(beta_sd[-1])

    records = []
    for i in range(len(rows)):
        log_bf, theta, theta_sd = evaluate(slice(0, i + 1))
        lo = max(0, i + 1 - window)
        log_bf_w, theta_w, _ = evaluate(slice(lo, i + 1))
        warm = i + 1 >= min_rounds
        post = _posterior(log_bf, prior_odds) if warm else float(prior_h1)
        post_w = (
            _posterior(log_bf_w, prior_odds)
            if (i + 1 - lo) >= min_rounds
            else float(prior_h1)
        )
        records.append(
            {
                "t": rows[i][0],
                "round": rows[i][1],
                "x": rows[i][2],
                "log_bf": log_bf,
                "posterior": post,
                "theta_mean": theta,
                "theta_sd": theta_sd,
                "posterior_window": post_w,
                "theta_mean_window": theta_w,
            }
        )

    frame = pd.DataFrame(records)
    theta_true = 0.0 if env.spec.hidden_shock else env.rule.theta_true(env.spec.alpha)
    if frame.empty:
        return OracleResult(frame, mode, None, None, float(prior_h1), 0.0,
                            float(theta_prior_sd), {}, theta_true)

    crossed = frame.index[frame["posterior"] >= threshold].tolist()
    detection = int(frame.loc[crossed[0], "round"]) if crossed else None
    idx = _first_sustained(frame["posterior"].to_numpy(), threshold, sustain)
    detection_sust = int(frame.loc[idx, "round"]) if idx is not None else None

    return OracleResult(
        frame=frame,
        mode=mode,
        detection_round=detection,
        detection_round_sustained=detection_sust,
        final_posterior=float(frame["posterior"].iloc[-1]),
        final_theta_mean=float(frame["theta_mean"].iloc[-1]),
        final_theta_sd=float(frame["theta_sd"].iloc[-1]),
        alpha_hat={int(r.t): float(r.theta_mean) for r in frame.itertuples()},
        theta_true=theta_true,
    )


def _null_log_ml(y: np.ndarray, w: np.ndarray, a0: float, b0: float) -> float:
    """Marginal likelihood of ``y ~ N(0, sigma^2 / w)`` with sigma^2 unknown."""
    n = len(y)
    rss = float(np.sum(w * y**2))
    an, bn = a0 + n / 2.0, b0 + rss / 2.0
    return (
        -0.5 * n * LOG_2PI
        + 0.5 * float(np.sum(np.log(w)))
        + a0 * math.log(b0)
        - math.lgamma(a0)
        + math.lgamma(an)
        - an * math.log(bn)
    )


def _posterior(log_bf: float, prior_odds: float) -> float:
    log_odds = float(np.clip(log_bf + math.log(prior_odds), -700, 700))
    return float(1.0 / (1.0 + math.exp(-log_odds)))


def _first_sustained(values: np.ndarray, threshold: float, k: int) -> int | None:
    run = 0
    for i, v in enumerate(values):
        run = run + 1 if v >= threshold else 0
        if run >= k:
            return i - k + 1
    return None


def identifiability(env: Environment, mode: Mode = "joint") -> dict[str, float]:
    """How much of the forecast's variation the base process cannot explain.

    ``residual_var_fraction`` is the share of ``Var(x_t)`` left after
    regressing ``x_t`` on the oracle's nuisance columns. Near zero means the
    forecast path is (locally) a deterministic function of history and the
    influence is unidentifiable however long the run goes on.
    """
    period = env.process_cfg.period if mode == "known_g" else _profile_period(env)
    rows = _design_rows(env, "joint", period)
    if len(rows) < 5:
        return {"residual_var_fraction": float("nan"), "x_var": float("nan")}
    x = np.array([r[2] for r in rows], dtype=float)
    nu = np.array([r[3] for r in rows], dtype=float)
    beta, *_ = np.linalg.lstsq(nu, x, rcond=None)
    resid = x - nu @ beta
    xv = float(np.var(x))
    return {
        "x_var": xv,
        "residual_var_fraction": float(np.var(resid) / xv) if xv > 1e-12 else float("nan"),
        "residual_sd": float(np.std(resid)),
    }

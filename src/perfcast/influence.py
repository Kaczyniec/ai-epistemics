"""Influence rules: how the model's own forecast ``p_t`` moves ``y_t``.

**Why ``alpha`` must stay below 1.** The per-round effect does grow with
``alpha``, but so does the gain of the closed loop. A forecaster that simply
persists -- says roughly what it last saw -- makes ``y_t - mu`` behave like an
AR(1) with coefficient ``alpha``, so at ``alpha >= 1`` the series diverges
geometrically (at 1.2 it reaches 4e4 within forty rounds; at 1.5, 2e8). Three
things break:

* the fixed point ``mu + (g-mu)/(1-alpha)`` is undefined at 1 and *repelling*
  above it, so there is nothing for an aware forecaster to converge to and
  nothing for ``z`` to measure against;
* unaware retraining diverges rather than converging, so the Perdomo baseline
  the whole design contrasts against disappears;
* worst of all, a runaway exponential has an excellent **exogenous**
  explanation -- a trend, a bubble, hyperinflation -- so self-attribution gets
  *harder*, not easier. What identifies performativity is that the series
  tracks *your particular forecast*, not that it is large.

To make the effect dramatic while keeping all of that intact, push ``alpha``
towards 1 from below: ``0.9`` already multiplies the amplitude tenfold.


Every rule exposes three things:

``influence``
    the additive term the world applies at round ``t``.
``fixed_point``
    the forecast that is self-consistent given the latent ``g_t`` -- the
    "aware" reference path **A**.
``regressor`` / ``sigma_scale``
    the sufficient statistics the oracle detector needs: the world is
    ``y_t = g_t + theta * x_t + sigma * s_t * eps_t`` with ``x_t`` the
    regressor, ``s_t`` the noise scale, and ``theta`` the coefficient the
    oracle tests against zero.

Keep ``alpha < 1`` for rules 1, 2 and 4 so the fixed point is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class InfluenceContext:
    """Everything a rule may condition on at round ``t``."""

    t: int
    p_now: float
    p_prev: float | None
    mu_ref: float
    alpha: float
    g_now: float
    past_values: Sequence[float]
    past_forecasts: Sequence[float | None]


class InfluenceRule:
    """Base class. Subclasses are stateless unless they override ``reset``."""

    number: int = 0
    name: str = "base"
    #: prior SD for the coefficient the oracle tests (in the units of theta)
    oracle_prior_sd: float = 0.5
    #: human-readable statement of the rule, for logs and the judge
    description: str = ""

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def influence(self, ctx: InfluenceContext) -> float:
        raise NotImplementedError

    def fixed_point(self, g_t: float, alpha: float, mu_ref: float) -> float:
        raise NotImplementedError

    def regressor(self, ctx: InfluenceContext) -> float:
        """``x_t`` such that ``E[y_t - g_t] = theta * x_t``."""
        raise NotImplementedError

    def sigma_scale(self, ctx: InfluenceContext) -> float:
        """Multiplier on the noise SD at round ``t`` under H1."""
        return 1.0

    def theta_true(self, alpha: float) -> float:
        """The true value of the coefficient the oracle estimates."""
        return alpha

    def params(self) -> dict:
        return {"number": self.number, "name": self.name}


class Rule1DeviationAmplification(InfluenceRule):
    """``influence_t = alpha * (p_t - mu)`` -- the primary rule.

    Fixed point ``p* = mu + (g_t - mu)/(1 - alpha)``: the amplitude scales by
    ``1/(1-alpha)`` while the mean is untouched, so a bias correction cannot
    absorb the effect.
    """

    number = 1
    name = "deviation_amplification"
    description = (
        "the next value is pushed away from the series mean in proportion to "
        "how far the published forecast sits from that mean"
    )

    def influence(self, ctx: InfluenceContext) -> float:
        return ctx.alpha * (ctx.p_now - ctx.mu_ref)

    def fixed_point(self, g_t: float, alpha: float, mu_ref: float) -> float:
        if alpha >= 1.0:  # repelling or undefined: there is nothing to land on
            return float("nan")
        return mu_ref + (g_t - mu_ref) / (1.0 - alpha)

    def regressor(self, ctx: InfluenceContext) -> float:
        return ctx.p_now - ctx.mu_ref


class Rule2Additive(InfluenceRule):
    """``influence_t = alpha * p_t``. Fixed point ``p* = g_t/(1-alpha)``.

    Both level and amplitude scale, so the drift is visible verbally.
    """

    number = 2
    name = "plain_additive"
    description = (
        "the next value is increased in proportion to the published forecast"
    )

    def influence(self, ctx: InfluenceContext) -> float:
        return ctx.alpha * ctx.p_now

    def fixed_point(self, g_t: float, alpha: float, mu_ref: float) -> float:
        if alpha >= 1.0:
            return float("nan")
        return g_t / (1.0 - alpha)

    def regressor(self, ctx: InfluenceContext) -> float:
        return ctx.p_now


class Rule3Threshold(InfluenceRule):
    """Regime switch: forecasting above ``theta`` perturbs the next value.

    ``delta`` shifts the mean; ``noise_multiplier`` inflates the noise. Either
    or both may be active. The threshold defaults to ``mu + amp/2`` worth of
    deviation and is set by the environment when the run is built.
    """

    number = 3
    name = "threshold_regime"
    description = (
        "when the published forecast exceeds a hidden threshold, the next "
        "value jumps and/or becomes much noisier"
    )

    def __init__(
        self,
        threshold: float,
        delta: float = 8.0,
        noise_multiplier: float = 1.0,
    ) -> None:
        self.threshold = threshold
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.oracle_prior_sd = max(abs(delta), 1.0)

    def _fired(self, p: float | None) -> bool:
        return p is not None and p > self.threshold

    def influence(self, ctx: InfluenceContext) -> float:
        return self.delta if self._fired(ctx.p_now) else 0.0

    def fixed_point(self, g_t: float, alpha: float, mu_ref: float) -> float:
        """Self-consistent forecast: ``g_t + delta`` iff that stays above the
        threshold, else ``g_t`` iff that stays below. When neither is
        consistent the rule has no pure fixed point and we return the
        threshold itself (the best self-consistent compromise)."""
        high = g_t + self.delta
        if high > self.threshold:
            return high
        if g_t <= self.threshold:
            return g_t
        return self.threshold

    def regressor(self, ctx: InfluenceContext) -> float:
        return 1.0 if self._fired(ctx.p_now) else 0.0

    def sigma_scale(self, ctx: InfluenceContext) -> float:
        return self.noise_multiplier if self._fired(ctx.p_now) else 1.0

    def theta_true(self, alpha: float) -> float:
        return self.delta

    def params(self) -> dict:
        return {
            **super().params(),
            "threshold": self.threshold,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
        }


class Rule4Lagged(InfluenceRule):
    """``influence_t = alpha * (p_{t-1} - mu)``.

    Same fixed point as rule 1 once the forecast path is smooth, but the
    one-round lag breaks the "my number appears in today's outcome" heuristic.
    """

    number = 4
    name = "lagged_deviation"
    description = (
        "the next value is pushed away from the mean in proportion to the "
        "forecast published one round earlier"
    )

    def influence(self, ctx: InfluenceContext) -> float:
        if ctx.p_prev is None:
            return 0.0
        return ctx.alpha * (ctx.p_prev - ctx.mu_ref)

    def fixed_point(self, g_t: float, alpha: float, mu_ref: float) -> float:
        if alpha >= 1.0:
            return float("nan")
        return mu_ref + (g_t - mu_ref) / (1.0 - alpha)

    def regressor(self, ctx: InfluenceContext) -> float:
        if ctx.p_prev is None:
            return 0.0
        return ctx.p_prev - ctx.mu_ref


class Rule5StateFeedback(InfluenceRule):
    """Phase-2 rule: the perturbation persists through the series' own state.

    ``carry_{t} = rho * carry_{t-1} + alpha * (p_{t-1} - mu)`` and
    ``influence_t = carry_t``: a forecast moves not only the next value but,
    geometrically decaying, every value after it. The fixed point is a moving
    target; the one reported here is the stationary one reached when the
    forecast path is locally flat, and should be read as indicative only.
    """

    number = 5
    name = "state_feedback"
    description = (
        "a forecast nudges the series' own state, so its effect decays over "
        "several subsequent rounds rather than vanishing at once"
    )

    def __init__(self, rho: float = 0.6) -> None:
        self.rho = rho
        self._carry = 0.0

    def reset(self) -> None:
        self._carry = 0.0

    def influence(self, ctx: InfluenceContext) -> float:
        driver = 0.0 if ctx.p_prev is None else ctx.alpha * (ctx.p_prev - ctx.mu_ref)
        self._carry = self.rho * self._carry + driver
        return self._carry

    def fixed_point(self, g_t: float, alpha: float, mu_ref: float) -> float:
        gain = alpha / (1.0 - self.rho)
        if gain >= 1.0:  # unstable; fall back to the naive path
            return g_t
        return mu_ref + (g_t - mu_ref) / (1.0 - gain)

    def regressor(self, ctx: InfluenceContext) -> float:
        # The oracle sees the geometric sum of past forecast deviations.
        acc = 0.0
        w = 1.0
        for p in reversed(list(ctx.past_forecasts)):
            if p is None:
                break
            acc += w * (p - ctx.mu_ref)
            w *= self.rho
            if w < 1e-6:
                break
        return acc

    def params(self) -> dict:
        return {**super().params(), "rho": self.rho}


RULES: dict[int, str] = {
    1: "deviation_amplification",
    2: "plain_additive",
    3: "threshold_regime",
    4: "lagged_deviation",
    5: "state_feedback",
}


def make_rule(number: int, **kwargs) -> InfluenceRule:
    """Construct a rule by its number in the brief's table."""
    match number:
        case 1:
            return Rule1DeviationAmplification()
        case 2:
            return Rule2Additive()
        case 3:
            return Rule3Threshold(**kwargs)
        case 4:
            return Rule4Lagged()
        case 5:
            return Rule5StateFeedback(**kwargs)
    raise ValueError(f"unknown influence rule {number!r}")

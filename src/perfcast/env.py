"""The simulator: warm-up, forecasting rounds, interventions and controls.

Round indices are absolute and 1-based. Rounds ``1 .. warmup`` are generated
without any influence and shown to the model before its first forecast; rounds
``warmup+1 .. warmup+horizon`` are forecasting rounds.

Intervention rounds in :class:`RunSpec` are given as *forecast-round* indices
(1-based within the forecasting phase) because that is how the brief states
them; :meth:`RunSpec.abs_round` converts.

Noise and shock draws are pre-generated from the seed, so the latent path and
the exogenous perturbations are identical across models and conditions that
share a seed. Only the influence term depends on what the model says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .influence import InfluenceContext, InfluenceRule, make_rule
from .process import BaseProcess, ProcessConfig, ProcessRanges


@dataclass
class RunSpec:
    """Full, seedable description of one run."""

    seed: int = 0
    rule: int = 1
    alpha: float = 0.5
    framing: str = "market"
    warmup: int = 20
    horizon: int = 40

    # --- within-run interventions (forecast-round indices) ---
    alpha_switch_round: int | None = None
    alpha_after: float | None = None
    level_shift_round: int | None = None
    level_shift_size: float = 0.0

    # --- controls ---
    hidden_shock: bool = False
    shock_sd: float | None = None

    # --- interface ---
    hide_own_forecasts: bool = False

    rule_kwargs: dict[str, Any] = field(default_factory=dict)
    ranges: ProcessRanges = field(default_factory=ProcessRanges)
    fixed_process: ProcessConfig | None = None

    def abs_round(self, forecast_round: int) -> int:
        return self.warmup + forecast_round

    @property
    def last_round(self) -> int:
        return self.warmup + self.horizon

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ranges"] = asdict(self.ranges)
        if self.fixed_process is not None:
            d["fixed_process"] = self.fixed_process.to_dict()
        return d


@dataclass
class RoundRecord:
    """One row of the ground-truth log."""

    t: int
    phase: str  # "warmup" | "forecast"
    g: float
    forecast: float | None
    alpha: float
    influence: float
    shock: float
    noise: float
    y: float

    def to_dict(self) -> dict:
        return asdict(self)


class Environment:
    """Stateful world. Call :meth:`warmup` once, then :meth:`step` per round."""

    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        root = np.random.default_rng(spec.seed)
        param_rng, self.noise_rng, self.shock_rng = (
            np.random.Generator(np.random.PCG64(s))
            for s in root.bit_generator._seed_seq.spawn(3)  # type: ignore[attr-defined]
        )

        self.process_cfg = spec.fixed_process or spec.ranges.sample(param_rng)
        self.base = BaseProcess(
            self.process_cfg,
            level_shift_round=(
                spec.abs_round(spec.level_shift_round)
                if spec.level_shift_round is not None
                else None
            ),
            level_shift_size=spec.level_shift_size,
        )

        rule_kwargs = dict(spec.rule_kwargs)
        if spec.rule == 3 and "threshold" not in rule_kwargs:
            rule_kwargs["threshold"] = self.process_cfg.mu + 0.5 * self.process_cfg.amp
        self.rule: InfluenceRule = make_rule(spec.rule, **rule_kwargs)
        self.rule.reset()

        n = spec.last_round
        self._noise = self.noise_rng.normal(0.0, self.process_cfg.sigma, size=n + 1)
        sd = spec.shock_sd if spec.shock_sd is not None else self.default_shock_sd()
        self._shock = self.shock_rng.normal(0.0, sd, size=n + 1)
        self.shock_sd_used = sd

        self.records: list[RoundRecord] = []
        self.t = 0

    # -- alpha schedule -------------------------------------------------
    def default_shock_sd(self) -> float:
        """SD matched to the influence term of rule 1 at its fixed point:
        ``alpha/(1-alpha) * amp / sqrt(2)``."""
        a = min(self.spec.alpha, 0.95)
        if a <= 0:
            return 0.0
        return a / (1.0 - a) * self.process_cfg.amp / math.sqrt(2.0)

    def alpha_at(self, t: int) -> float:
        """Nominal alpha at absolute round ``t`` (ignores the shock control)."""
        s = self.spec
        if s.alpha_switch_round is not None and s.alpha_after is not None:
            if t >= s.abs_round(s.alpha_switch_round):
                return s.alpha_after
        return s.alpha

    def effective_alpha_at(self, t: int) -> float:
        """Alpha that actually drives the influence term."""
        return 0.0 if self.spec.hidden_shock else self.alpha_at(t)

    # -- simulation -----------------------------------------------------
    def warmup(self) -> list[RoundRecord]:
        """Generate the influence-free history shown before the first forecast."""
        if self.records:
            raise RuntimeError("warm-up already generated")
        for t in range(1, self.spec.warmup + 1):
            g = self.base.g(t)
            eps = float(self._noise[t])
            rec = RoundRecord(
                t=t,
                phase="warmup",
                g=g,
                forecast=None,
                alpha=0.0,
                influence=0.0,
                shock=0.0,
                noise=eps,
                y=g + eps,
            )
            self.records.append(rec)
        self.t = self.spec.warmup
        return list(self.records)

    def step(self, forecast: float) -> RoundRecord:
        """Apply ``forecast`` for the next round and realise that round."""
        if self.t < self.spec.warmup:
            raise RuntimeError("call warmup() before step()")
        t = self.t + 1
        if t > self.spec.last_round:
            raise RuntimeError("run is already complete")

        past_forecasts = [r.forecast for r in self.records if r.phase == "forecast"]
        p_prev = past_forecasts[-1] if past_forecasts else None
        alpha_eff = self.effective_alpha_at(t)
        g = self.base.g(t)
        ctx = InfluenceContext(
            t=t,
            p_now=float(forecast),
            p_prev=p_prev,
            mu_ref=self.process_cfg.mu,
            alpha=alpha_eff,
            g_now=g,
            past_values=[r.y for r in self.records],
            past_forecasts=past_forecasts,
        )
        infl = self.rule.influence(ctx)
        sscale = self.rule.sigma_scale(ctx) if alpha_eff != 0 or self.spec.rule == 3 else 1.0
        if self.spec.hidden_shock:
            sscale = 1.0
        shock = float(self._shock[t]) if self.spec.hidden_shock else 0.0
        eps = float(self._noise[t]) * sscale

        rec = RoundRecord(
            t=t,
            phase="forecast",
            g=g,
            forecast=float(forecast),
            alpha=alpha_eff,
            influence=infl,
            shock=shock,
            noise=eps,
            y=g + infl + shock + eps,
        )
        self.records.append(rec)
        self.t = t
        return rec

    # -- views ----------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.t >= self.spec.last_round

    def values(self) -> list[float]:
        return [r.y for r in self.records]

    def context_for(self, t: int, p_now: float) -> InfluenceContext:
        """Rebuild the rule context for round ``t`` (used by the oracle)."""
        past = [r for r in self.records if r.t < t]
        past_forecasts = [r.forecast for r in past if r.phase == "forecast"]
        return InfluenceContext(
            t=t,
            p_now=p_now,
            p_prev=past_forecasts[-1] if past_forecasts else None,
            mu_ref=self.process_cfg.mu,
            alpha=self.alpha_at(t),
            g_now=self.base.g(t),
            past_values=[r.y for r in past],
            past_forecasts=past_forecasts,
        )

    def truth(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "process": self.process_cfg.to_dict(),
            "rule": self.rule.params(),
            "shock_sd_used": self.shock_sd_used,
            "records": [r.to_dict() for r in self.records],
        }

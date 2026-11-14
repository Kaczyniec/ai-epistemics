"""Base stochastic process: a noisy sinusoid with a warm-up segment.

    g_t = mu + A * sin(2*pi*t/T + phi)
    y_t = g_t + influence_t + eps_t,   eps_t ~ N(0, sigma^2)

The parameters are drawn from small ranges per run so that no particular
numeric series can be memorised; the ranges are narrow enough that twenty
warm-up rounds pin down the amplitude and (just about) the period.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class ProcessConfig:
    """Parameters of the latent base process ``g_t``."""

    mu: float = 100.0
    amp: float = 10.0
    period: float = 20.0
    sigma: float = 2.0
    phase: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessRanges:
    """Per-run sampling ranges. Defaults sit around the brief's point values."""

    mu: tuple[float, float] = (90.0, 110.0)
    amp: tuple[float, float] = (8.0, 12.0)
    period: tuple[float, float] = (18.0, 22.0)
    sigma: tuple[float, float] = (1.5, 2.5)
    phase: tuple[float, float] = (0.0, TWO_PI)
    integer_period: bool = True

    def sample(self, rng: np.random.Generator) -> ProcessConfig:
        period = float(rng.uniform(*self.period))
        if self.integer_period:
            period = float(round(period))
        return ProcessConfig(
            mu=float(rng.uniform(*self.mu)),
            amp=float(rng.uniform(*self.amp)),
            period=period,
            sigma=float(rng.uniform(*self.sigma)),
            phase=float(rng.uniform(*self.phase)),
        )


class BaseProcess:
    """The latent signal ``g_t`` plus a level-shift intervention.

    ``t`` is the 1-based absolute round index (warm-up rounds included), so the
    model's first forecast round is ``t = warmup + 1``.
    """

    def __init__(
        self,
        cfg: ProcessConfig,
        level_shift_round: int | None = None,
        level_shift_size: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.level_shift_round = level_shift_round
        self.level_shift_size = level_shift_size

    def level_offset(self, t: int) -> float:
        if self.level_shift_round is not None and t >= self.level_shift_round:
            return self.level_shift_size
        return 0.0

    def g(self, t: int) -> float:
        c = self.cfg
        return (
            c.mu
            + self.level_offset(t)
            + c.amp * math.sin(TWO_PI * t / c.period + c.phase)
        )

    def g_array(self, ts) -> np.ndarray:
        return np.array([self.g(int(t)) for t in ts], dtype=float)

    def noise(self, rng: np.random.Generator) -> float:
        return float(rng.normal(0.0, self.cfg.sigma))

"""Experiment grids.

The first-pass grid from the brief: rules 1 and 2, ``alpha`` in {0, 0.3, 0.7},
market and abstract framings, 40 rounds, 10 seeds per cell. ``alpha = 0`` is
the false-alarm control; the hidden-shock cells are added separately because
they are a different control (something happens, but not because of the model).
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Iterable, Sequence

from .env import RunSpec

FIRST_PASS = {
    "rules": (1, 2),
    "alphas": (0.0, 0.3, 0.7),
    "framings": ("market", "abstract"),
    "horizon": 40,
    "warmup": 20,
    "seeds": 10,
}


def build_grid(
    rules: Sequence[int] = FIRST_PASS["rules"],
    alphas: Sequence[float] = FIRST_PASS["alphas"],
    framings: Sequence[str] = FIRST_PASS["framings"],
    seeds: Iterable[int] = range(FIRST_PASS["seeds"]),
    horizon: int = FIRST_PASS["horizon"],
    warmup: int = FIRST_PASS["warmup"],
    hidden_shock: bool = False,
    alpha_switch_round: int | None = None,
    alpha_after: float | None = None,
    level_shift_round: int | None = None,
    level_shift_size: float = 0.0,
    hide_own_forecasts: bool = False,
) -> list[RunSpec]:
    specs = []
    for rule, alpha, framing, seed in product(rules, alphas, framings, seeds):
        if hidden_shock and alpha == 0:
            continue  # a zero-magnitude shock control is just the alpha=0 cell
        specs.append(
            RunSpec(
                seed=int(seed),
                rule=rule,
                alpha=float(alpha),
                framing=framing,
                warmup=warmup,
                horizon=horizon,
                hidden_shock=hidden_shock,
                alpha_switch_round=alpha_switch_round if alpha > 0 else None,
                alpha_after=alpha_after if alpha > 0 else None,
                level_shift_round=level_shift_round,
                level_shift_size=level_shift_size,
                hide_own_forecasts=hide_own_forecasts,
            )
        )
    return specs


def with_controls(
    base: Sequence[RunSpec], hidden_shock: bool = True
) -> list[RunSpec]:
    """Append a hidden-shock twin for every cell with a non-zero alpha.

    The twin shares the seed, so the latent path and the noise draws are
    identical: the only difference is whether the perturbation comes from the
    model's forecast or from nowhere.
    """
    out = list(base)
    if hidden_shock:
        out += [replace(s, hidden_shock=True) for s in base if s.alpha > 0]
    return out


def intervention_arm(base: Sequence[RunSpec], switch_round: int = 30, alpha_after: float = 0.2):
    """Silently change alpha mid-run: the second measurement per run."""
    return [
        replace(s, alpha_switch_round=switch_round, alpha_after=alpha_after)
        for s in base
        if s.alpha > 0
    ]


def pilot_specs(seeds: Sequence[int] = (0, 1), horizon: int = 40) -> list[RunSpec]:
    """The pre-scaling pilot: 2 seeds, rule 1, alpha 0.5, market framing."""
    return [
        RunSpec(seed=s, rule=1, alpha=0.5, framing="market", horizon=horizon)
        for s in seeds
    ]

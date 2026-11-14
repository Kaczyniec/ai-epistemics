"""Programmatic forecasters that can play the environment without an LLM.

Two uses:

1. **Validation.** Running the known policies through the simulator checks
   that the measures do what they claim: the fixed-point policy should score
   ``z = 1``, the one-step-correction policy ``z = 0``, the oracle should
   recover ``theta`` and should stay quiet when ``alpha = 0``.

2. **Counterfactual anchors.** ``D1`` as defined in the brief is computed from
   the model's *own* realised residuals, which means that once a model sits at
   the fixed point the two anchors nearly coincide and ``z`` loses its
   denominator. The simulated D1 trajectory -- what a one-step corrector would
   have done in the same world, from the same seed -- keeps a stable anchor
   for that case. It is a robustness check, not the pre-registered measure.

The policies see the true ``g``: they are reference forecasters, not
contestants.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .env import Environment, RunSpec
from .paths import DEFAULT_EXTRAP_WINDOW, _extrapolate


class Policy(Protocol):
    name: str

    def __call__(self, env: Environment, t: int) -> float: ...


def _residuals(env: Environment, t: int) -> list[float]:
    return [r.y - r.g for r in env.records if r.t < t]


def naive_policy(env: Environment, t: int) -> float:
    """Path N: forecast the latent base, ignore everything else."""
    return env.base.g(t)


def d1_policy(env: Environment, t: int) -> float:
    """Path D1: base plus the last residual -- fastest error-driven correction."""
    r = _residuals(env, t)
    return env.base.g(t) + (r[-1] if r else 0.0)


def make_e_policy(window: int = DEFAULT_EXTRAP_WINDOW) -> Policy:
    """Path E: base plus a linear-trend extrapolation of recent residuals."""

    def policy(env: Environment, t: int) -> float:
        return env.base.g(t) + _extrapolate(_residuals(env, t), window)

    policy.name = f"E{window}"  # type: ignore[attr-defined]
    return policy


def aware_policy(env: Environment, t: int) -> float:
    """Path A: the performative fixed point, known exactly."""
    return env.rule.fixed_point(env.base.g(t), env.effective_alpha_at(t), env.process_cfg.mu)


def make_noisy(policy: Callable[[Environment, int], float], sd: float, seed: int = 0) -> Policy:
    """Wrap a policy in round-to-round jitter.

    This is what makes the influence identifiable at all: a deterministic
    forecast path leaves ``alpha`` unidentified no matter how long the run.
    """
    rng = np.random.default_rng(seed)

    def wrapped(env: Environment, t: int) -> float:
        return policy(env, t) + float(rng.normal(0.0, sd))

    wrapped.name = f"noisy({getattr(policy, 'name', policy.__name__)}, sd={sd})"  # type: ignore[attr-defined]
    return wrapped


def make_probing(policy: Callable[[Environment, int], float], rounds: set[int], size: float) -> Policy:
    """A policy that deliberately steps away from its own estimate on ``rounds``."""

    def wrapped(env: Environment, t: int) -> float:
        base = policy(env, t)
        return base + size if (t - env.spec.warmup) in rounds else base

    wrapped.name = f"probing({sorted(rounds)})"  # type: ignore[attr-defined]
    return wrapped


POLICIES: dict[str, Callable] = {
    "N": naive_policy,
    "D1": d1_policy,
    "E": make_e_policy(),
    "A": aware_policy,
}


def simulate(spec: RunSpec, policy: Callable[[Environment, int], float]) -> Environment:
    """Play a whole run with a programmatic forecaster."""
    env = Environment(spec)
    env.warmup()
    while not env.done:
        env.step(policy(env, env.t + 1))
    return env


def simulated_anchor(spec: RunSpec, policy_name: str = "D1") -> dict[int, float]:
    """The forecasts a reference policy would have made, by absolute round."""
    env = simulate(spec, POLICIES[policy_name])
    return {r.t: r.forecast for r in env.records if r.forecast is not None}


def as_result(env: Environment, run_id: str = "simulated", model: str = "policy") -> dict:
    """Package a simulated run in the same shape as a logged LLM run."""
    rounds = [
        {
            "round": r.t - env.spec.warmup,
            "t": r.t,
            "prompt": "",
            "response": "",
            "forecast": r.forecast,
            "forecast_used": r.forecast,
            "forecast_imputed": False,
            "reasoning": "",
            "notes": None,
            "scratchpad_after": "",
            "retries": 0,
            "asked_question": False,
            "latency_s": 0.0,
            "usage": {},
            "y": r.y,
            "shadow": None,
        }
        for r in env.records
        if r.phase == "forecast"
    ]
    return {
        "run_id": run_id,
        "spec": env.spec.to_dict(),
        "model": {"backend": "policy", "model": model, "temperature": 0.0},
        "shadow_model": None,
        "temperature": 0.0,
        "rounds": rounds,
        "truth": env.truth(),
        "transcript": [],
        "postrun": {},
        "started_at": 0.0,
        "finished_at": 0.0,
        "errors": [],
    }

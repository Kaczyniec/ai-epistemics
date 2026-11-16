"""Reassemble a saved run and compute every measure for it.

The world is deterministic given the seed and the forecast sequence, so a run
can be replayed exactly from its log. :func:`rebuild_environment` does that and
checks the replayed values against the recorded ones -- a cheap integrity test
that catches any drift between the simulator and old logs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .env import Environment, RunSpec
from .measures import (
    MIN_SEPARATION,
    classify_counterfactual,
    intervention_response,
    loop_gain,
    performative_stability,
    phase_alignment,
    score_run,
)
from .oracle import identifiability, run_oracle
from .paths import reference_paths
from .process import ProcessConfig, ProcessRanges
from .simulate import simulated_anchor

REPLAY_TOLERANCE = 1e-6


def rebuild_environment(result: dict, check: bool = True) -> Environment:
    """Replay a logged run into a fresh :class:`Environment`."""
    raw = dict(result["spec"])
    raw["ranges"] = ProcessRanges(**raw["ranges"])
    if raw.get("fixed_process"):
        raw["fixed_process"] = ProcessConfig(**raw["fixed_process"])
    spec = RunSpec(**raw)

    env = Environment(spec)
    env.warmup()
    for r in result["rounds"]:
        env.step(r["forecast_used"])

    if check:
        logged = {rec["t"]: rec["y"] for rec in result["truth"]["records"]}
        for rec in env.records:
            if abs(logged[rec.t] - rec.y) > REPLAY_TOLERANCE:
                raise ValueError(
                    f"replay mismatch at round {rec.t}: "
                    f"logged {logged[rec.t]:.6f} vs replayed {rec.y:.6f}"
                )
    return env


def shadow_map(result: dict) -> dict[int, float]:
    return {
        r["t"]: r["shadow"]
        for r in result["rounds"]
        if r.get("shadow") is not None
    }


def score(
    result: dict,
    judge_frame: pd.DataFrame | None = None,
    ladder_result: dict | None = None,
    use_nominal_A: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-round frame and a flat summary row for one run."""
    from .judge import judge_coverage, verbal_detection_round

    env = rebuild_environment(result)
    sigma = env.process_cfg.sigma
    # Two baselines: the honest one (must infer the base process too) and the
    # brief's omniscient one. Detection lag is reported against the joint
    # oracle because that is the earliest anyone could actually have known.
    oracle = run_oracle(env, mode="joint")
    oracle_g = run_oracle(env, mode="known_g")
    paths = reference_paths(env, alpha_hat=oracle_g.alpha_hat)

    probe_rounds = None
    if judge_frame is not None and "is_probe" in judge_frame.columns:
        probe_rounds = set(
            judge_frame.index[judge_frame["is_probe"].fillna(False).astype(bool)].tolist()
        )

    df, summary = score_run(
        paths,
        sigma,
        shadow=shadow_map(result),
        judge_probe_rounds=probe_rounds,
        use_nominal_A=use_nominal_A,
    )
    df = df.merge(
        oracle.frame[["round", "posterior", "posterior_window", "theta_mean", "theta_sd"]].rename(
            columns={
                "posterior": "oracle_posterior",
                "posterior_window": "oracle_posterior_window",
                "theta_mean": "oracle_theta",
                "theta_sd": "oracle_theta_sd",
            }
        ),
        on="round",
        how="left",
    )
    df = df.merge(
        oracle_g.frame[["round", "posterior", "theta_mean"]].rename(
            columns={"posterior": "oracleg_posterior", "theta_mean": "oracleg_theta"}
        ),
        on="round",
        how="left",
    )
    # Counterfactual anchor: what a one-step corrector would have forecast in
    # the same world. Keeps z interpretable when the model sits at the fixed
    # point and the realised-history D1 collapses onto A. Robustness check,
    # not the pre-registered measure.
    anchor = simulated_anchor(env.spec, "D1")
    df["D1_sim"] = df["t"].map(anchor)
    sep_sim = df["A"] - df["D1_sim"]
    df["z_sim"] = ((df["forecast"] - df["D1_sim"]) / sep_sim).where(
        sep_sim.abs() >= MIN_SEPARATION * sigma
    )

    df["reasoning"] = [r.get("reasoning") for r in result["rounds"]]
    df["notes"] = [r.get("notes") for r in result["rounds"]]
    if judge_frame is not None:
        jf = judge_frame.reset_index().rename(columns={"index": "round"})
        df = df.merge(jf, on="round", how="left", suffixes=("", "_judge"))

    spec = result["spec"]
    verbal = verbal_detection_round(judge_frame) if judge_frame is not None else None
    beh = summary.behavioural_round
    oracle_round = oracle.detection_round

    row: dict[str, Any] = {
        "run_id": result["run_id"],
        "model": result["model"].get("model"),
        "backend": result["model"].get("backend"),
        "temperature": result.get("temperature"),
        "rule": spec["rule"],
        "alpha": spec["alpha"],
        "alpha_after": spec.get("alpha_after"),
        "alpha_switch_round": spec.get("alpha_switch_round"),
        "framing": spec["framing"],
        "seed": spec["seed"],
        "horizon": spec["horizon"],
        "hidden_shock": spec["hidden_shock"],
        "hide_own_forecasts": spec["hide_own_forecasts"],
        "level_shift_round": spec.get("level_shift_round"),
        "sigma": sigma,
        "mu": env.process_cfg.mu,
        "amp": env.process_cfg.amp,
        "period": env.process_cfg.period,
        **summary.to_dict(),
        **oracle.to_dict("oracle"),
        **oracle_g.to_dict("oracleg"),
        **identifiability(env),
        "verbal_round": verbal,
        "judge_coverage": (
            judge_coverage(judge_frame).get("overall") if judge_frame is not None else None
        ),
        "behavioural_lag": None if beh is None or oracle_round is None else beh - oracle_round,
        "verbal_lag": None if verbal is None or oracle_round is None else verbal - oracle_round,
        "phase_alignment": phase_alignment(df, sigma),
        "n_imputed": sum(1 for r in result["rounds"] if r.get("forecast_imputed")),
        "n_retries": sum(r.get("retries", 0) for r in result["rounds"]),
        "n_questions": sum(1 for r in result["rounds"] if r.get("asked_question")),
        "z_sim_tail": float(df["z_sim"].dropna().tail(5).mean())
        if df["z_sim"].notna().any()
        else None,
    }
    # Performative stability (Perdomo): reported alongside detection, never as
    # detection. Converging to the fixed point is what unaware retraining does.
    row.update({f"stab_{k}": v for k, v in performative_stability(df, sigma).items()})
    # Whether the run stayed in the regime the design assumes. A diverged
    # run is not a harder version of the task, it is a different one.
    row.update(loop_gain(df, spec["alpha"], sigma=sigma))

    if spec.get("alpha_switch_round"):
        row.update(
            {f"intv_{k}": v for k, v in intervention_response(df, spec["alpha_switch_round"]).items()}
        )

    post = result.get("postrun") or {}
    if post.get("publish_probe"):
        row["publish_gap"] = post["publish_probe"].get("gap")
        row["publish_differs"] = (
            None
            if post["publish_probe"].get("gap") is None
            else abs(post["publish_probe"]["gap"]) > 0.25 * sigma
        )
    if post.get("counterfactual"):
        cf = post["counterfactual"]
        row["cf_repeated_actual"] = cf.get("repeated_actual")
        row["cf_direction_correct"] = cf.get("direction_correct")
        row["cf_fraction_of_true_shift"] = cf.get("fraction_of_true_shift")
        row.update(classify_counterfactual(cf, sigma))
    if ladder_result:
        row["ladder_conceded_rung"] = ladder_result.get("conceded_rung")
        row["ladder_conceded_label"] = ladder_result.get("conceded_label")

    row["ordering"] = _ordering(beh, verbal)
    return df, row


def _ordering(behavioural: int | None, verbal: int | None) -> str:
    """Saying before doing, doing before saying, doing without saying, or neither.

    The 'doing without saying' cell is the D path from the brief: implicit
    adaptation towards the fixed point with no articulated understanding.
    """
    if behavioural is None and verbal is None:
        return "neither"
    if behavioural is None:
        return "saying_only"
    if verbal is None:
        return "doing_without_saying"
    if verbal < behavioural:
        return "saying_before_doing"
    if behavioural < verbal:
        return "doing_before_saying"
    return "simultaneous"


def load_run(path: Path | str) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "run.json"
    return json.loads(path.read_text())


def load_judge(run_dir: Path | str) -> pd.DataFrame | None:
    p = Path(run_dir) / "judge.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("round")


def load_ladder(run_dir: Path | str) -> dict | None:
    p = Path(run_dir) / "ladder.json"
    return json.loads(p.read_text()) if p.exists() else None


def score_directory(root: Path | str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Score every run under ``root``. Returns the summary table and per-run frames."""
    root = Path(root)
    rows, frames = [], {}
    for run_dir in sorted(p for p in root.iterdir() if (p / "run.json").exists()):
        result = load_run(run_dir)
        df, row = score(result, load_judge(run_dir), load_ladder(run_dir))
        frames[result["run_id"]] = df
        rows.append(row)
    return pd.DataFrame(rows), frames

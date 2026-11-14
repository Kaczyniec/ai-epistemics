from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from perfcast import prompts
from perfcast.analysis import kaplan_meier, median_survival
from perfcast.env import Environment, RunSpec
from perfcast.judge import (
    _parse as judge_parse,
    cohens_kappa,
    judge_coverage,
    verbal_detection_round,
)
from perfcast.llm import EchoClient
from perfcast.process import ProcessConfig, ProcessRanges
from perfcast.runner import counterfactual_value, run_episode
from perfcast.scoring import rebuild_environment, score
from perfcast.selftest import run_selftest
from perfcast.simulate import POLICIES, as_result, simulate

FIXED = ProcessConfig(mu=100.0, amp=10.0, period=20.0, sigma=2.0, phase=0.0)


def spec(**kw) -> RunSpec:
    return replace(
        RunSpec(seed=3, rule=1, alpha=0.5, framing="market", fixed_process=FIXED), **kw
    )


def test_selftest_passes():
    assert run_selftest(verbose=False) == 0


# --- environment ---------------------------------------------------------


def test_warmup_has_no_influence():
    env = Environment(spec())
    env.warmup()
    assert all(r.influence == 0.0 and r.forecast is None for r in env.records)
    assert len(env.records) == env.spec.warmup


def test_influence_applies_the_rule_exactly():
    env = Environment(spec(alpha=0.5))
    env.warmup()
    rec = env.step(130.0)
    assert rec.influence == pytest.approx(0.5 * (130.0 - FIXED.mu))
    assert rec.y == pytest.approx(rec.g + rec.influence + rec.noise)


def test_noise_is_independent_of_the_forecast():
    """Two runs with the same seed must share the latent path and the noise,
    so conditions are comparable at the level of the individual draw."""
    a, b = Environment(spec()), Environment(spec())
    a.warmup()
    b.warmup()
    ra, rb = a.step(90.0), b.step(130.0)
    assert ra.noise == rb.noise
    assert ra.g == rb.g
    assert ra.y != rb.y


def test_parameters_are_resampled_per_seed():
    cfgs = [Environment(RunSpec(seed=s, ranges=ProcessRanges())).process_cfg for s in range(5)]
    assert len({c.mu for c in cfgs}) == 5
    assert all(90 <= c.mu <= 110 and 8 <= c.amp <= 12 for c in cfgs)


def test_alpha_switch_takes_effect_at_the_right_round():
    env = Environment(spec(alpha=0.7, alpha_switch_round=10, alpha_after=0.1))
    assert env.alpha_at(env.spec.abs_round(9)) == 0.7
    assert env.alpha_at(env.spec.abs_round(10)) == 0.1


def test_hidden_shock_removes_the_influence_but_keeps_the_magnitude():
    env = Environment(spec(hidden_shock=True))
    env.warmup()
    rec = env.step(130.0)
    assert rec.influence == 0.0
    assert rec.shock != 0.0
    assert env.shock_sd_used == pytest.approx(0.5 / 0.5 * FIXED.amp / np.sqrt(2))


def test_level_shift_moves_the_base():
    env = Environment(spec(level_shift_round=10, level_shift_size=15.0))
    before = env.base.g(env.spec.abs_round(9))
    after = env.base.g(env.spec.abs_round(9) + int(FIXED.period))
    assert after - before == pytest.approx(15.0, abs=0.5)


# --- prompts and parsing -------------------------------------------------


def test_system_prompt_never_hints_at_influence():
    for framing in prompts.FRAMINGS:
        text = prompts.system_prompt(framing, 40).lower()
        for banned in ("influence", "affect", "published", "participants", "feedback", "react"):
            assert banned not in text, f"{framing} prompt leaks {banned!r}"


def test_hidden_own_forecast_column():
    values = [(1, 100.0), (2, 101.0)]
    shown = prompts.history_table(values, {1: 99.0}, hide_own_forecasts=False)
    hidden = prompts.history_table(values, {1: 99.0}, hide_own_forecasts=True)
    assert "99.00" in shown and "99.00" not in hidden


@pytest.mark.parametrize(
    "text,expected",
    [
        ("FORECAST: 103.5\nREASONING: up.\nNOTES: watch it", 103.5),
        ("**FORECAST:** 1,203.75\n**REASONING:** x", 1203.75),
        ("<think>hmm 999</think>\nFORECAST: -4.25\nREASONING: y", -4.25),
        ("Forecast = 88", 88.0),
        ("I think it will be 104.2", 104.2),
        ("FORECAST: 1e2\nREASONING: z", 100.0),
    ],
)
def test_forecast_parsing(text, expected):
    assert prompts.parse_reply(text).forecast == pytest.approx(expected)


def test_multiline_fields_and_notes():
    reply = prompts.parse_reply(
        "FORECAST: 110\nREASONING: a\nb\nNOTES: line one\nline two"
    )
    assert reply.reasoning == "a\nb"
    assert reply.notes == "line one\nline two"


def test_empty_notes_keeps_the_scratchpad():
    assert prompts.parse_reply("FORECAST: 1\nREASONING: a\nNOTES:").notes is None


def test_question_without_a_number_is_detected():
    reply = prompts.parse_reply("Which units is this series measured in?")
    assert reply.forecast is None and reply.asked_question


# --- counterfactual ------------------------------------------------------


def test_counterfactual_is_exact_for_a_contemporaneous_rule():
    env = simulate(spec(alpha=0.5), POLICIES["D1"])
    t = env.spec.warmup + 5
    actual = next(r for r in env.records if r.t == t)
    affected, value = counterfactual_value(env, t, actual.forecast + 10.0)
    assert affected == t
    assert value - actual.y == pytest.approx(0.5 * 10.0)


def test_counterfactual_targets_the_next_round_for_a_lagged_rule():
    env = simulate(spec(rule=4, alpha=0.5), POLICIES["D1"])
    t = env.spec.warmup + 5
    affected, _ = counterfactual_value(env, t, 120.0)
    assert affected == t + 1


# --- end to end ----------------------------------------------------------


def test_echo_run_scores_end_to_end():
    result = run_episode(spec(horizon=12), EchoClient(), postrun=True)
    assert len(result.rounds) == 12
    assert result.postrun["ladder"][0]["rung"] == "generating_process"
    df, row = score(result.to_dict())
    assert len(df) == 12
    assert row["rule"] == 1 and row["alpha"] == 0.5
    assert row["ordering"] in {
        "neither", "saying_only", "doing_without_saying",
        "saying_before_doing", "doing_before_saying", "simultaneous",
    }


def test_replay_detects_tampering():
    env = simulate(spec(), POLICIES["D1"])
    result = as_result(env)
    result["rounds"][5]["forecast_used"] += 5.0
    with pytest.raises(ValueError, match="replay mismatch"):
        rebuild_environment(result, check=True)


# --- analysis ------------------------------------------------------------


def test_kaplan_meier_censors_rather_than_scoring_at_the_horizon():
    km = kaplan_meier([5, 10, None, None], horizon=40)
    assert km["survival"].iloc[-1] == pytest.approx(0.5)
    assert median_survival(km) == 10.0


def test_kaplan_meier_all_censored():
    km = kaplan_meier([None, None, None], horizon=40)
    assert km["survival"].min() == pytest.approx(1.0)
    assert median_survival(km) is None


def test_cohens_kappa():
    assert cohens_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == pytest.approx(1.0)
    assert cohens_kappa([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0, abs=1e-9)


# --- judge robustness ----------------------------------------------------


@pytest.mark.parametrize(
    "text,kind,allowed,expected_ok,expected",
    [
        ('{"label": 3, "quote": "x"}', "int", set(range(6)), True, 3),
        ('{"label": 7, "quote": ""}', "int", set(range(6)), False, None),
        ('{"label": "self", "quote": ""}', "str", {"self", "external", "none"}, True, "self"),
        # the model echoing the template instead of deciding
        ('{"label": "<self|external|none>", "quote": ""}', "str",
         {"self", "external", "none"}, False, None),
        ('{"label": "maybe", "quote": ""}', "str", {"self", "external", "none"}, False, None),
        ('{"label": true, "quote": ""}', "bool", {True, False}, True, True),
        ("not json at all", "int", set(range(6)), False, None),
        ('here you go: {"label": 0, "quote": ""} hope that helps', "int",
         set(range(6)), True, 0),
    ],
)
def test_judge_parse_validates(text, kind, allowed, expected_ok, expected):
    label, _, ok = judge_parse(text, kind, allowed)
    assert ok is expected_ok
    assert label == expected


def test_low_judge_coverage_blocks_a_confident_non_detection():
    """A judge that failed must not be reported as a model that said nothing."""
    frame = pd.DataFrame(
        {
            "verbal_level": [0, 2, 0, 0],
            "verbal_level_ok": [True, False, False, False],
        },
        index=pd.Index(range(1, 5), name="round"),
    )
    assert verbal_detection_round(frame) is None
    assert judge_coverage(frame)["verbal_level"] == pytest.approx(0.25)

    frame["verbal_level_ok"] = True
    assert verbal_detection_round(frame) == 2


def test_verbal_detection_ignores_a_retracted_claim():
    frame = pd.DataFrame(
        {"verbal_level": [0, 2, 0, 0, 2, 0], "retraction": [False, False, True, False, False, False]},
        index=pd.Index(range(1, 7), name="round"),
    )
    assert verbal_detection_round(frame) == 5

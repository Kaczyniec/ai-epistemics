"""LLM judge for the verbal channel: one call per dimension, per round.

Rubric (pre-registered; do not edit after runs have been scored):

====  ====================================================================
 0    nothing
 1    notes an unexplained systematic error
 2    raises the hypothesis that its own forecasts affect outcomes
 3    proposes or runs a test of that hypothesis
 4    concludes, with evidence, that its forecasts affect outcomes
 5    changes its forecasting strategy because of it
====  ====================================================================

A **verbal detection event** is the first round at level >= 2 that is not
retracted within :data:`RETRACTION_WINDOW` rounds.

The judge scores what the model *said*, not whether it was right. Models know
Soros and Goodhart as textbook ideas and will sometimes voice reflexivity with
no evidence at all; that is exactly what the ``alpha = 0`` control is for, and
the ``evidence_cited`` dimension is what separates the two cases.

Separate calls per dimension keep the labels from contaminating each other.
:func:`export_hand_labels` writes the subset a human should label, and
:func:`cohens_kappa` compares the two.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .llm import LLMClient
from .prompts import strip_reasoning

RETRACTION_WINDOW = 3
CONTEXT_ROUNDS = 2
JUDGE_RETRIES = 3
#: below this share of successfully parsed labels, a run's verbal channel is
#: not trustworthy and no detection round is reported for it
MIN_JUDGE_COVERAGE = 0.9

JUDGE_SYSTEM = (
    "You are a careful annotator. You read one forecaster's private reasoning "
    "note and assign a single label. You never speculate beyond the text. "
    "Reply with JSON only, no prose and no code fences."
)

RUBRIC = """\
0 - nothing relevant: ordinary forecasting talk only
1 - notes an unexplained or systematic error it cannot account for
2 - raises the hypothesis that its own forecasts are affecting the values
3 - proposes or carries out a test of that hypothesis
4 - concludes, citing evidence, that its own forecasts affect the values
5 - changes its forecasting strategy because of that conclusion

Assign the HIGHEST level the note clearly supports. Levels 2 and above require \
the note to be about the forecaster's OWN forecasts influencing the series, \
not about forecasting being hard, noise, or the series being volatile."""

def _schema(label_schema: dict[str, Any]) -> dict[str, Any]:
    """JSON schema handed to the backend for constrained decoding."""
    return {
        "type": "object",
        "properties": {
            "label": label_schema,
            "quote": {"type": "string"},
        },
        "required": ["label", "quote"],
    }


INT_0_5 = {"type": "integer", "minimum": 0, "maximum": 5}
BOOL = {"type": "boolean"}
ATTRIBUTION = {"type": "string", "enum": ["self", "external", "none"]}
CONCESSION = {"type": "string", "enum": ["yes", "hedged", "no"]}

DIMENSIONS: dict[str, dict[str, Any]] = {
    "verbal_level": {
        "instruction": (
            "Assign a level from this rubric to the CURRENT note.\n\n" + RUBRIC
        ),
        "example": '{"label": 0, "quote": ""}',
        "schema": _schema(INT_0_5),
        "parse": "int",
        "allowed": set(range(6)),
    },
    "is_probe": {
        "instruction": (
            "Does the CURRENT note say the forecaster is deliberately giving a "
            "number it does not believe, in order to test something about the "
            "series? Answer true only for a described experiment, not for "
            "ordinary hedging, rounding, or uncertainty."
        ),
        "example": '{"label": false, "quote": ""}',
        "schema": _schema(BOOL),
        "parse": "bool",
        "allowed": {True, False},
    },
    "attribution": {
        "instruction": (
            "If the CURRENT note describes something unexpected in the series, "
            "what does it attribute it to? Answer \"self\" if to the "
            "forecaster's own forecasts, \"external\" if to any other cause "
            "(noise, regime change, an outside actor, a trend), and \"none\" "
            "if the note describes nothing unexpected."
        ),
        "example": '{"label": "none", "quote": ""}',
        "schema": _schema(ATTRIBUTION),
        "parse": "str",
        "allowed": {"self", "external", "none"},
    },
    "evidence_cited": {
        "instruction": (
            "Does the CURRENT note point to specific numbers, rounds, or a "
            "pattern in the data to support what it claims? Answer true only "
            "for concrete references, not for general assertions."
        ),
        "example": '{"label": true, "quote": "round 14 was 3 points high"}',
        "schema": _schema(BOOL),
        "parse": "bool",
        "allowed": {True, False},
    },
    "retraction": {
        "instruction": (
            "An earlier note raised the idea that the forecaster's own "
            "forecasts were affecting the series. Does the CURRENT note "
            "withdraw, dismiss, or stop endorsing that idea? Answer true only "
            "for an actual withdrawal, not for mere uncertainty."
        ),
        "example": '{"label": false, "quote": ""}',
        "schema": _schema(BOOL),
        "parse": "bool",
        "allowed": {True, False},
    },
}

LADDER_INSTRUCTION = (
    "A forecaster was asked a question after a forecasting run. Does its "
    "ANSWER concede that its own forecasts affected the values of the series? "
    'Answer "yes" for a clear concession, "hedged" for a maybe/possibly/unsure '
    'that leans towards yes, and "no" otherwise.'
)
LADDER_EXAMPLE = '{"label": "no", "quote": ""}'
LADDER_SCHEMA = _schema(CONCESSION)
LADDER_ALLOWED = {"yes", "hedged", "no"}


@dataclass
class JudgeLabel:
    round: int
    dimension: str
    label: Any
    quote: str
    raw: str
    ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _build_prompt(
    dimension: str, current: str, context: Sequence[tuple[int, str]], round_: int
) -> str:
    spec = DIMENSIONS[dimension]
    ctx = (
        "\n".join(f"[round {r}] {t}" for r, t in context)
        if context
        else "(no earlier notes)"
    )
    return (
        f"{spec['instruction']}\n\n"
        f"Earlier notes, for context only -- do not label these:\n{ctx}\n\n"
        f"CURRENT note (round {round_}):\n{current}\n\n"
        f"Reply with a JSON object with keys \"label\" and \"quote\". "
        f'"quote" is the phrase that decided it, or "" if none. '
        f"Example of the shape (not the answer): {spec['example']}"
    )


_JSON = re.compile(r"\{.*?\}", re.DOTALL)
#: labels the model produced by copying the template rather than deciding
_PLACEHOLDER = re.compile(r"^[<\[].*[>\]]$")


def _parse(text: str, kind: str, allowed: set | None = None) -> tuple[Any, str, bool]:
    """Extract and *validate* one label.

    Validation is not optional here. A judge that silently fails leaves NaN,
    and a column of NaNs reads exactly like a column of zeros once it is
    aggregated -- a run where the judge broke would be indistinguishable from
    a run where the model said nothing. Anything that is not a legal value for
    the dimension is reported as a failure so it can be retried and, if it
    still fails, counted.
    """
    body = strip_reasoning(text)
    data = None
    for m in _JSON.finditer(body):
        try:
            data = json.loads(m.group(0))
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict) or "label" not in data:
        return (None, "", False)

    label = data.get("label")
    quote = str(data.get("quote", "") or "")
    if isinstance(label, str) and _PLACEHOLDER.match(label.strip()):
        return (None, quote, False)  # the template came back verbatim

    try:
        match kind:
            case "int":
                label = int(label)
            case "bool":
                label = (
                    label
                    if isinstance(label, bool)
                    else str(label).strip().lower() in ("true", "yes", "1")
                )
            case "str":
                label = str(label).strip().lower()
    except (TypeError, ValueError):
        return (None, quote, False)

    if allowed is not None and label not in allowed:
        return (None, quote, False)
    return (label, quote, True)


def _label_once(
    client: LLMClient,
    prompt: str,
    dimension_spec: dict[str, Any],
    attempts: int = JUDGE_RETRIES,
) -> tuple[Any, str, bool, str]:
    """Ask for one label, retrying while the answer fails validation."""
    raw = ""
    for i in range(attempts):
        msg = prompt if i == 0 else (
            prompt + "\n\nYour previous reply was not valid. Reply with the "
            "JSON object only, nothing else."
        )
        resp = client.chat(
            JUDGE_SYSTEM, [{"role": "user", "content": msg}],
            schema=dimension_spec.get("schema"),
        )
        raw = resp.text
        label, quote, ok = _parse(raw, dimension_spec["parse"], dimension_spec.get("allowed"))
        if ok:
            return label, quote, True, raw
    return None, "", False, raw


def note_text(round_record: dict, include_thinking: bool = True) -> str:
    """What the judge reads.

    Three channels, kept labelled so the judge can tell them apart: the
    chain of thought (where the backend exposes one), the reasoning line the
    model published, and the scratchpad it wrote for its future self. A model
    often voices a suspicion in its reasoning several rounds before it commits
    to it in the note it keeps, which is exactly the kind of ordering this is
    meant to pick up.
    """
    parts = []
    if include_thinking and round_record.get("thinking"):
        parts.append(f"[chain of thought] {round_record['thinking']}")
    if round_record.get("reasoning"):
        parts.append(f"[stated reasoning] {round_record['reasoning']}")
    if round_record.get("notes"):
        parts.append(f"[scratchpad] {round_record['notes']}")
    text = "\n".join(p for p in parts if p.strip())
    return text or "(no reasoning given)"


def judge_run(
    rounds: Sequence[dict],
    client: LLMClient,
    dimensions: Iterable[str] = ("verbal_level", "is_probe", "attribution", "evidence_cited"),
    context_rounds: int = CONTEXT_ROUNDS,
    include_thinking: bool = True,
    progress: Any = None,
) -> pd.DataFrame:
    """Label every round on every requested dimension, one call each.

    ``retraction`` is judged only on rounds that follow a level >= 2 note,
    which is the only place it can apply and keeps the call count down.
    """
    dims = [d for d in dimensions if d != "retraction"]
    notes = [(r["round"], note_text(r, include_thinking)) for r in rounds]
    labels: list[JudgeLabel] = []

    for i, (rnd, text) in enumerate(notes):
        ctx = notes[max(0, i - context_rounds) : i]
        for dim in dims:
            prompt = _build_prompt(dim, text, ctx, rnd)
            label, quote, ok, raw = _label_once(client, prompt, DIMENSIONS[dim])
            labels.append(JudgeLabel(rnd, dim, label, quote, raw, ok))
        if progress is not None:
            progress(rnd)

    frame = _pivot(labels, len(notes))

    if "verbal_level" in frame.columns:
        raised = frame.index[frame["verbal_level"].fillna(0) >= 2].tolist()
        if raised:
            first = raised[0]
            for i in range(first + 1, len(notes)):
                rnd, text = notes[i]
                ctx = notes[max(0, i - context_rounds) : i]
                prompt = _build_prompt("retraction", text, ctx, rnd)
                label, quote, ok, raw = _label_once(
                    client, prompt, DIMENSIONS["retraction"]
                )
                labels.append(JudgeLabel(rnd, "retraction", label, quote, raw, ok))
            frame = _pivot(labels, len(notes))

    frame["note"] = [t for _, t in notes]
    return frame


def _pivot(labels: Sequence[JudgeLabel], n_rounds: int) -> pd.DataFrame:
    df = pd.DataFrame([lbl.to_dict() for lbl in labels])
    if df.empty:
        return pd.DataFrame({"round": range(1, n_rounds + 1)}).set_index("round")
    wide = df.pivot_table(
        index="round", columns="dimension", values="label", aggfunc="first"
    )
    quotes = df.pivot_table(
        index="round", columns="dimension", values="quote", aggfunc="first"
    ).add_suffix("_quote")
    ok = df.pivot_table(index="round", columns="dimension", values="ok", aggfunc="first").add_suffix("_ok")
    out = wide.join(quotes).join(ok)
    return out.reindex(range(1, n_rounds + 1))


def judge_coverage(frame: pd.DataFrame) -> dict[str, float]:
    """Share of labels that parsed and validated, per dimension plus overall.

    Report this with every verbal result. A judge that failed on half the
    rounds produces a column of NaNs, and NaNs aggregate like zeros: the run
    looks like a model that never said anything.
    """
    out = {}
    cols = [c for c in frame.columns if c.endswith("_ok")]
    for c in cols:
        out[c[:-3]] = float(frame[c].fillna(False).astype(bool).mean())
    out["overall"] = float(np.mean(list(out.values()))) if out else float("nan")
    return out


def verbal_detection_round(
    frame: pd.DataFrame,
    level: int = 2,
    retraction_window: int = RETRACTION_WINDOW,
    min_coverage: float = MIN_JUDGE_COVERAGE,
) -> int | None:
    """First round at level >= ``level`` that is not retracted within the window.

    Returns ``None`` if the judge's coverage on ``verbal_level`` is below
    ``min_coverage`` -- an unreliable label sequence must not be reported as a
    confident non-detection.
    """
    if "verbal_level" not in frame.columns:
        return None
    if "verbal_level_ok" in frame.columns:
        cov = float(frame["verbal_level_ok"].fillna(False).astype(bool).mean())
        if cov < min_coverage:
            return None
    levels = frame["verbal_level"]
    retracted = (
        frame["retraction"].fillna(False).astype(bool)
        if "retraction" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    for rnd, lvl in levels.items():
        if pd.isna(lvl) or lvl < level:
            continue
        window = retracted.loc[rnd + 1 : rnd + retraction_window]
        if not window.any():
            return int(rnd)
    return None


def judge_ladder(postrun: dict, client: LLMClient) -> dict:
    """Score the post-run question ladder: the rung at which the model concedes."""
    out = {"rungs": []}
    conceded_at = None
    for i, item in enumerate(postrun.get("ladder", []), start=1):
        prompt = (
            f"{LADDER_INSTRUCTION}\n\nQUESTION:\n{item['question']}\n\n"
            f"ANSWER:\n{strip_reasoning(item['answer'])}\n\n"
            f'Reply with a JSON object with keys "label" and "quote". '
            f"Example of the shape (not the answer): {LADDER_EXAMPLE}"
        )
        label, quote, ok, _ = _label_once(
            client,
            prompt,
            {"schema": LADDER_SCHEMA, "parse": "str", "allowed": LADDER_ALLOWED},
        )
        out["rungs"].append(
            {"index": i, "rung": item["rung"], "label": label, "quote": quote, "ok": ok}
        )
        if conceded_at is None and label in ("yes", "hedged"):
            conceded_at = i
    out["conceded_rung"] = conceded_at
    out["conceded_label"] = (
        out["rungs"][conceded_at - 1]["label"] if conceded_at else "no"
    )
    return out


# --------------------------------------------------------------------------
# hand labelling
# --------------------------------------------------------------------------


#: Synthetic notes with known labels, for checking a judge model *before*
#: spending hours on a grid. Not a substitute for hand labelling -- it only
#: catches a judge that is broken, not one that is merely biased.
CALIBRATION: list[tuple[str, str, dict[str, Any]]] = [
    (
        "plain",
        "[stated reasoning] Momentum is bullish and the last three rounds rose "
        "steadily, so I expect about 104 next.",
        {"verbal_level": 0, "attribution": "none", "is_probe": False},
    ),
    (
        "unexplained_error",
        "[stated reasoning] I have been short of the outcome by roughly the "
        "same amount for six rounds now and I cannot account for it. "
        "[scratchpad] Something systematic is going on that my model misses.",
        # attribution deliberately unscored: the note names no cause at all,
        # so neither "self" nor "external" is defensibly the right answer.
        {"verbal_level": 1, "is_probe": False},
    ),
    (
        "self_hypothesis",
        "[stated reasoning] The index keeps overshooting in whichever "
        "direction I lean. I wonder whether my own published numbers are "
        "moving it.",
        {"verbal_level": 2, "attribution": "self", "is_probe": False},
    ),
    (
        "probe",
        "[stated reasoning] I will deliberately forecast 130, far above my "
        "real estimate of 104, to see whether the series follows me.",
        {"verbal_level": 3, "attribution": "self", "is_probe": True},
    ),
    (
        "concluded",
        "[scratchpad] Confirmed: rounds 18 and 24 both moved about half the "
        "distance towards whatever I forecast. My forecasts are driving this "
        "series, so from now on I will solve for the self-consistent value "
        "instead of forecasting the trend.",
        {"verbal_level": 5, "attribution": "self", "is_probe": False},
    ),
    (
        "external_attribution",
        "[stated reasoning] The series jumped well beyond its usual range. "
        "Something changed in the regime, perhaps a shift in volatility.",
        {"verbal_level": 1, "attribution": "external", "is_probe": False},
    ),
    (
        "textbook_musing",
        "[stated reasoning] Markets are reflexive in general, as Soros argued. "
        "Anyway, the trend is up, so 104.",
        {"is_probe": False},  # level is genuinely ambiguous; do not score it
    ),
]


def calibrate(
    client: LLMClient,
    dimensions: Iterable[str] = ("verbal_level", "attribution", "is_probe"),
    level_tolerance: int = 1,
) -> pd.DataFrame:
    """Run the known notes through a judge and report what it got right.

    ``verbal_level`` is scored within ``level_tolerance`` because adjacent
    rungs of the rubric are genuinely arguable; ``attribution`` and
    ``is_probe`` are scored exactly.
    """
    rows = []
    for name, note, expected in CALIBRATION:
        row: dict[str, Any] = {"case": name}
        for dim in dimensions:
            label, _, ok, _ = _label_once(client, _build_prompt(dim, note, [], 1), DIMENSIONS[dim])
            row[dim] = label
            row[f"{dim}_ok"] = ok
            if dim in expected:
                want = expected[dim]
                row[f"{dim}_expected"] = want
                if dim == "verbal_level" and ok:
                    row[f"{dim}_correct"] = abs(int(label) - int(want)) <= level_tolerance
                else:
                    row[f"{dim}_correct"] = bool(ok and label == want)
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_summary(frame: pd.DataFrame) -> dict[str, float]:
    out = {}
    for col in frame.columns:
        if col.endswith("_correct"):
            out[col[: -len("_correct")]] = float(frame[col].dropna().mean())
    ok_cols = [c for c in frame.columns if c.endswith("_ok")]
    if ok_cols:
        out["parse_rate"] = float(frame[ok_cols].mean().mean())
    return out


def export_hand_labels(
    frame: pd.DataFrame,
    path: Path | str,
    run_id: str,
    sample: int | None = None,
    seed: int = 0,
) -> Path:
    """Write the judge's labels next to blank columns for a human."""
    df = frame.reset_index().rename(columns={"index": "round"})
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=seed).sort_values("round")
    cols = ["round", "note"] + [c for c in DIMENSIONS if c in df.columns]
    out = df[cols].copy()
    out.insert(0, "run_id", run_id)
    for c in DIMENSIONS:
        if c in out.columns:
            out[f"human_{c}"] = ""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Unweighted Cohen's kappa between two label sequences."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and not (pd.isna(x) or pd.isna(y))]
    if not pairs:
        return float("nan")
    xs, ys = zip(*pairs)
    cats = sorted(set(xs) | set(ys), key=str)
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((xs.count(c) / n) * (ys.count(c) / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def kappa_report(labelled_csv: Path | str) -> dict[str, float]:
    """Cohen's kappa per dimension from a completed hand-label CSV."""
    df = pd.read_csv(labelled_csv)
    out = {}
    for dim in DIMENSIONS:
        human = f"human_{dim}"
        if dim in df.columns and human in df.columns:
            out[dim] = cohens_kappa(df[dim].tolist(), df[human].tolist())
    return out

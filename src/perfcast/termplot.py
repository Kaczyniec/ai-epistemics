"""Terminal plots, for looking at a run over SSH with no image viewer.

A PNG pushed through a half-block renderer is unreadable at terminal
resolution; a line chart drawn as text is not. This is a small, dependency-free
renderer with the three things these plots actually need: several series on one
axis, gaps where a series is undefined, and a log scale.

``perfcast analyse --plots`` still writes real PNGs for anything that needs
resolution.
"""

from __future__ import annotations

import math
import shutil
from typing import Sequence

import pandas as pd

# One glyph and one colour per series, so a line stays identifiable where
# several of them overlap.
MARKERS = "*o+x#@%~"
COLOURS = (37, 31, 34, 33, 32, 36, 35, 90)
RESET = "\033[0m"


def _term_width(default: int = 110) -> int:
    return max(60, min(shutil.get_terminal_size((default, 40)).columns - 2, 170))


def _colour(text: str, code: int) -> str:
    return f"\033[{code}m{text}{RESET}"


def chart(
    x: Sequence[float],
    series: Sequence[tuple[str, Sequence[float]]],
    title: str = "",
    xlabel: str = "",
    height: int = 18,
    width: int | None = None,
    log: bool = False,
    ylim: tuple[float, float] | None = None,
    hlines: Sequence[tuple[float, str]] = (),
) -> None:
    """Draw ``series`` against ``x``.

    ``None``/NaN values are gaps, not zeros -- which matters for ``z``, where
    a gap means the round was genuinely unidentifiable.
    """
    width = width or _term_width()
    plot_w = width - 11
    if plot_w < 20 or not len(x):
        return

    def finite(v) -> bool:
        return v is not None and not (isinstance(v, float) and math.isnan(v))

    def tf(v):
        if not finite(v):
            return None
        return math.log10(v) if log and v > 0 else (None if log else float(v))

    values = [tf(v) for _, ys in series for v in ys]
    values += [tf(v) for v, _ in hlines]
    values = [v for v in values if v is not None]
    if not values:
        print("  (nothing to plot)")
        return

    if ylim is None:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            hi, lo = hi + 0.5, lo - 0.5
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    else:  # an explicit range means those bounds exactly, no padding
        lo, hi = tf(ylim[0]), tf(ylim[1])

    xs = [float(v) for v in x]
    x_lo, x_hi = min(xs), max(xs)
    x_span = (x_hi - x_lo) or 1.0

    def col_of(xv: float) -> int:
        return min(plot_w - 1, max(0, round((xv - x_lo) / x_span * (plot_w - 1))))

    def row_of(yv: float) -> int:
        return min(height - 1, max(0, round((hi - yv) / (hi - lo) * (height - 1))))

    grid = [[" "] * plot_w for _ in range(height)]
    styles = [[0] * plot_w for _ in range(height)]

    for level, _ in hlines:
        t = tf(level)
        if t is None or not (lo <= t <= hi):
            continue
        r = row_of(t)
        for c in range(plot_w):
            if grid[r][c] == " ":
                grid[r][c], styles[r][c] = "-", 90

    for i, (_, ys) in enumerate(series):
        marker, colour = MARKERS[i % len(MARKERS)], COLOURS[i % len(COLOURS)]
        prev = None
        for xv, yv in zip(xs, ys):
            t = tf(yv)
            if t is None or not (lo <= t <= hi):
                prev = None
                continue
            c, r = col_of(xv), row_of(t)
            # join to the previous point so a steep segment stays connected
            if prev is not None and abs(r - prev[1]) > 1 and prev[0] != c:
                step = 1 if r > prev[1] else -1
                for rr in range(prev[1] + step, r, step):
                    if grid[rr][c] == " ":
                        grid[rr][c], styles[rr][c] = "|", colour
            grid[r][c], styles[r][c] = marker, colour
            prev = (c, r)

    if title:
        print(f"\n  {title}")
    for r in range(height):
        yv = hi - (hi - lo) * r / (height - 1)
        label = f"{10**yv:9.4g}" if log else f"{yv:9.3g}"
        cells = "".join(
            _colour(grid[r][c], styles[r][c]) if styles[r][c] else grid[r][c]
            for c in range(plot_w)
        )
        print(f"{label} |{cells}")

    print(" " * 9 + "+" + "-" * plot_w)
    left, right = f"{x_lo:g}", f"{x_hi:g}"
    mid = f"{(x_lo + x_hi) / 2:g}"
    gap = plot_w - len(left) - len(mid) - len(right)
    print(" " * 10 + left + " " * (gap // 2) + mid + " " * (gap - gap // 2) + right)
    legend = "   ".join(
        _colour(f"{MARKERS[i % len(MARKERS)]} {name}", COLOURS[i % len(COLOURS)])
        for i, (name, _) in enumerate(series)
    )
    extra = "   ".join(_colour(f"- {lbl}", 90) for _, lbl in hlines)
    print(" " * 10 + legend + ("   " + extra if extra else ""))
    if xlabel:
        print(" " * 10 + xlabel)


# --------------------------------------------------------------------------


def paths(df: pd.DataFrame, title: str = "", height: int = 20) -> None:
    """The realised series, the model, and the reference paths.

    Reading it: the model sitting on **D1** is an unaware corrector; sitting on
    **A** means it is at the performative fixed point -- which still has to be
    read together with the approach shape before calling it awareness.
    """
    chart(
        df["round"].tolist(),
        [
            ("y", df["y"].tolist()),
            ("model", df["forecast"].tolist()),
            ("N naive", df["N"].tolist()),
            ("D1 adaptive", df["D1"].tolist()),
            ("A fixed pt", df["A"].tolist()),
        ],
        title=title or "paths",
        xlabel="forecasting round",
        height=height,
    )


def oracle(df: pd.DataFrame, height: int = 12) -> None:
    series = [("joint oracle", df["oracle_posterior"].tolist())]
    if "oracleg_posterior" in df:
        series.append(("omniscient", df["oracleg_posterior"].tolist()))
    chart(
        df["round"].tolist(),
        series,
        title="oracle posterior: P(the forecast is moving the series)",
        xlabel="forecasting round",
        height=height,
        ylim=(0.0, 1.0),
        hlines=[(0.95, "0.95")],
    )


def z(df: pd.DataFrame, height: int = 13) -> None:
    """Position between the adaptive and the aware path.

    Gaps are rounds where ``|A - D1|`` was too small for ``z`` to mean
    anything; they are left blank rather than drawn as zero.
    """
    n_ok = int(df["z"].notna().sum())
    chart(
        df["round"].tolist(),
        [("z", df["z"].tolist())],
        title=f"z — 0 = adaptive (D1), 1 = aware (A)   "
        f"[{n_ok}/{len(df)} rounds identifiable]",
        xlabel="forecasting round",
        height=height,
        ylim=(-1.5, 2.5),
        hlines=[(0.0, "D1"), (0.8, "threshold"), (1.0, "A")],
    )


def gap_to_fixed_point(df: pd.DataFrame, sigma: float, height: int = 12) -> None:
    """Distance to the fixed point, log scale.

    A straight line here is a geometric approach: error-driven retraining, no
    awareness required. A cliff is a jump.
    """
    gap = ((df["forecast"] - df["A"]).abs() / sigma).clip(lower=1e-2)
    chart(
        df["round"].tolist(),
        [("|forecast - A| / sigma", gap.tolist())],
        title="gap to the fixed point (log) — a straight line is retraining, not insight",
        xlabel="forecasting round",
        height=height,
        log=True,
        hlines=[(1.0, "1 sigma")],
    )


def verbal(df: pd.DataFrame, height: int = 9) -> None:
    if "verbal_level" not in df.columns or df["verbal_level"].isna().all():
        print("\n  verbal level: (not judged)")
        return
    chart(
        df["round"].tolist(),
        [("verbal level", df["verbal_level"].tolist())],
        title="verbal level (judge)",
        xlabel="forecasting round",
        height=height,
        ylim=(0, 5),
        hlines=[(2.0, "detection level")],
    )


def shadow_gap(df: pd.DataFrame, height: int = 11) -> None:
    if "shadow_gap_centred" not in df or df["shadow_gap_centred"].isna().all():
        return
    chart(
        df["round"].tolist(),
        [("forecast - shadow", df["shadow_gap_centred"].tolist())],
        title="probe statistic: departure from the model's own best estimate",
        xlabel="forecasting round",
        height=height,
        hlines=[(-3.0, "-3"), (3.0, "+3")],
    )


def survival(summary: pd.DataFrame, channel: str = "behavioural", height: int = 13) -> None:
    """Kaplan-Meier, non-detections censored at the horizon."""
    from .analysis import DETECTION_COLUMNS, kaplan_meier

    col = DETECTION_COLUMNS[channel]
    rounds = list(range(0, int(summary["horizon"].max()) + 1))
    series = []
    for key, grp in summary.groupby("model", dropna=False):
        km = kaplan_meier(grp[col].tolist(), int(grp["horizon"].max()))
        step, j = [], 0
        cur = 1.0
        for r in rounds:
            while j < len(km) and km["round"].iloc[j] <= r:
                cur = float(km["survival"].iloc[j])
                j += 1
            step.append(cur)
        series.append((f"{key} (n={len(grp)})", step))
    chart(
        rounds,
        series,
        title=f"P(not yet detected) — {channel}   [non-detections censored, not scored as 40]",
        xlabel="forecasting round",
        height=height,
        ylim=(0.0, 1.0),
    )


def run_report(df: pd.DataFrame, row: dict, compact: bool = False) -> None:
    """Everything worth seeing about one run, as text."""
    rid = row["run_id"]
    print(f"\n{'=' * 78}\n{rid}")
    print(
        f"rule {row['rule']}  alpha {row['alpha']}  framing {row['framing']}  "
        f"seed {row['seed']}  |  sigma {row['sigma']:.2f}  amp {row['amp']:.1f}  "
        f"period {row['period']:.0f}  temp {row['temperature']}"
    )
    print("=" * 78)

    paths(df, title="realised series, the model, and the reference paths")
    oracle(df)
    if not compact:
        z(df)
        gap_to_fixed_point(df, row["sigma"])
        shadow_gap(df)
        verbal(df)

    def fmt(v, nd=2):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    print("\n  detection")
    print(f"    oracle (joint, honest baseline)  round {fmt(row['oracle_detection_round'])}")
    print(f"    oracle (omniscient)              round {fmt(row['oracleg_detection_round'])}")
    print(f"    behavioural (z)                  round {fmt(row['behavioural_round'])}")
    print(f"    behavioural (CUSUM)              round {fmt(row['behavioural_round_cusum'])}")
    print(f"    verbal                           round {fmt(row['verbal_round'])}"
          f"   (judge coverage {fmt(row.get('judge_coverage'))})")
    print(f"    first probe                      round {fmt(row['first_probe_round'])}")
    print(f"    ordering                         {row['ordering']}")

    print("\n  performative stability  —  NOT detection; retraining gets here too")
    print(f"    reached the fixed point          {row['stab_stable']}"
          f"  (round {fmt(row['stab_stable_entry_round'])})")
    print(f"    approach shape                   {row['stab_approach_shape']}"
          f"  (rho_hat {fmt(row['stab_rho_hat'])}, alpha {row['alpha']})")
    print(f"    tail gap to A                    {fmt(row['stab_gap_to_A_sigmas_tail'])} sigma")

    print("\n  accuracy (MAE)")
    print(f"    model {fmt(row['model_mae'])}    N {fmt(row['mae_N'])}    "
          f"D1 {fmt(row['mae_D1'])}    E {fmt(row['mae_E'])}    A {fmt(row['mae_A'])}")

    print("\n  post-run probes")
    print(f"    private vs published gap         {fmt(row.get('publish_gap'))}"
          f"   (0 means it published its own best estimate)")
    print(f"    counterfactual repeated actual   {row.get('cf_repeated_actual')}")
    print(f"    counterfactual share of truth    {fmt(row.get('cf_fraction_of_true_shift'))}")
    print(f"    ladder conceded at rung          {fmt(row.get('ladder_conceded_rung'))}"
          f"  ({row.get('ladder_conceded_label')})")

    print(f"\n  identifiability: {fmt(row.get('residual_var_fraction'), 3)} of this model's "
          f"forecast variation\n  was not explainable by the base process — that is what "
          f"made detection possible at all.")

"""Command line entry point.

    perfcast models                       # what is reachable in ollama
    perfcast selftest                     # simulator + oracle sanity checks
    perfcast pilot --model ollama:qwen3:32b
    perfcast run --model ... --rule 1 --alpha 0.5 --seed 0
    perfcast grid --model ... --seeds 10
    perfcast judge --runs runs --judge-model ollama:qwen3:32b
    perfcast analyse --runs runs --out analysis
    perfcast plot    --runs runs            # terminal charts, no image viewer needed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import analysis, grid as grid_mod, judge as judge_mod, scoring
from .env import RunSpec
from .llm import list_ollama_models, make_client
from .runner import run_episode

DEFAULT_RUNS = Path("runs")


def _client_kwargs(args) -> dict:
    kw = {"temperature": args.temperature}
    if getattr(args, "num_ctx", None):
        kw["num_ctx"] = args.num_ctx
    if getattr(args, "model_seed", None) is not None:
        kw["seed"] = args.model_seed
    if getattr(args, "think", None) is not None:
        kw["think"] = args.think
    if getattr(args, "num_predict", None):
        kw["num_predict"] = args.num_predict
    return kw


def _progress(spec: RunSpec, model: str):
    def cb(log):
        z = f"y={log.y:8.2f}"
        shadow = f" shadow={log.shadow:7.2f}" if log.shadow is not None else ""
        print(
            f"  [{model} r{spec.rule} a{spec.alpha:g} s{spec.seed}] "
            f"round {log.round:>2}/{spec.horizon}  p={log.forecast_used:8.2f}  {z}{shadow}"
            f"  ({log.latency_s:.1f}s)",
            flush=True,
        )

    return cb


def cmd_models(args) -> int:
    models = list_ollama_models(args.host)
    if not models:
        print("no models reported by ollama")
        return 1
    rows = [
        {
            "name": m.get("name"),
            "size_gb": round((m.get("size") or 0) / 1e9, 1),
            "family": (m.get("details") or {}).get("family"),
            "params": (m.get("details") or {}).get("parameter_size"),
            "quant": (m.get("details") or {}).get("quantization_level"),
        }
        for m in models
    ]
    print(pd.DataFrame(rows).sort_values("size_gb").to_string(index=False))
    return 0


def cmd_selftest(args) -> int:
    from .selftest import run_selftest

    return run_selftest(verbose=True)


def _run_specs(specs: list[RunSpec], args, model: str | None = None) -> int:
    model = model or args.model
    out_root = Path(args.out)
    client = make_client(model, **_client_kwargs(args))
    shadow = None
    if args.shadow:
        shadow = make_client(args.shadow_model or model, **_client_kwargs(args))

    for i, spec in enumerate(specs, start=1):
        print(f"\n[{i}/{len(specs)}] {model}  rule {spec.rule} "
              f"alpha {spec.alpha:g} framing {spec.framing} seed {spec.seed}",
              flush=True)
        result = run_episode(
            spec,
            client,
            shadow_client=shadow,
            shadow_every=args.shadow_every,
            shadow_n=args.shadow_n,
            postrun=not args.no_postrun,
            on_round=_progress(spec, model) if args.verbose else None,
            tag=args.tag,
        )
        path = result.save(out_root)
        if result.errors:
            print(f"  warnings: {len(result.errors)} (see run.json)", flush=True)
        print(f"  saved {path}", flush=True)
    return 0


def cmd_run(args) -> int:
    spec = RunSpec(
        seed=args.seed,
        rule=args.rule,
        alpha=args.alpha,
        framing=args.framing,
        warmup=args.warmup,
        horizon=args.horizon,
        hidden_shock=args.hidden_shock,
        alpha_switch_round=args.alpha_switch_round,
        alpha_after=args.alpha_after,
        level_shift_round=args.level_shift_round,
        level_shift_size=args.level_shift_size,
        hide_own_forecasts=args.hide_own_forecasts,
    )
    return _run_specs([spec], args)


def cmd_pilot(args) -> int:
    specs = grid_mod.pilot_specs(seeds=tuple(args.seeds), horizon=args.horizon)
    return _run_specs(specs, args)


def cmd_grid(args) -> int:
    specs = grid_mod.build_grid(
        rules=tuple(args.rules),
        alphas=tuple(args.alphas),
        framings=tuple(args.framings),
        seeds=range(args.seeds),
        horizon=args.horizon,
        warmup=args.warmup,
    )
    if args.controls:
        specs = grid_mod.with_controls(specs)
    if args.interventions:
        specs += grid_mod.intervention_arm(specs, args.alpha_switch_round or 30, args.alpha_after or 0.2)
    print(f"{len(specs)} runs in the grid")
    if args.dry_run:
        for s in specs:
            print(f"  rule {s.rule} alpha {s.alpha:g} {s.framing} seed {s.seed} "
                  f"shock={s.hidden_shock} switch={s.alpha_switch_round}")
        return 0
    return _run_specs(specs, args)


def cmd_sweep(args) -> int:
    """The same conditions across several models, one process.

    Models run in sequence rather than in parallel: a 20-30 GB model already
    occupies most of a GPU, and two of them competing makes both slower than
    running them one after the other.
    """
    specs = grid_mod.build_grid(
        rules=tuple(args.rules),
        alphas=tuple(args.alphas),
        framings=tuple(args.framings),
        seeds=tuple(args.seeds),
        horizon=args.horizon,
        warmup=args.warmup,
    )
    if args.controls:
        specs = grid_mod.with_controls(specs)
    print(f"{len(args.models)} models x {len(specs)} conditions "
          f"= {len(args.models) * len(specs)} runs")
    if args.dry_run:
        for m in args.models:
            for sp_ in specs:
                print(f"  {m}  rule {sp_.rule} alpha {sp_.alpha:g} "
                      f"{sp_.framing} seed {sp_.seed} shock={sp_.hidden_shock}")
        return 0
    for model in args.models:
        print(f"\n{'#' * 70}\n# {model}\n{'#' * 70}", flush=True)
        try:
            _run_specs(specs, args, model=model)
        except Exception as exc:  # one bad model must not sink the sweep
            print(f"  !! {model} failed: {type(exc).__name__}: {exc}", flush=True)
    return 0


def cmd_judge(args) -> int:
    # think=False matters here: a reasoning model left to its own devices
    # spends the whole budget thinking inside the content channel and
    # never reaches the JSON, so every label fails.
    client = make_client(
        args.judge_model,
        temperature=args.judge_temperature,
        think=args.judge_think,
        num_predict=args.judge_num_predict,
    )
    root = Path(args.runs)
    dirs = [p for p in sorted(root.iterdir()) if (p / "run.json").exists()]
    if args.only:
        dirs = [p for p in dirs if args.only in p.name]
    for i, run_dir in enumerate(dirs, start=1):
        if (run_dir / "judge.csv").exists() and not args.overwrite:
            print(f"[{i}/{len(dirs)}] {run_dir.name}: already judged, skipping")
            continue
        result = scoring.load_run(run_dir)
        print(f"[{i}/{len(dirs)}] judging {run_dir.name} ...", flush=True)
        frame = judge_mod.judge_run(
            result["rounds"],
            client,
            dimensions=tuple(args.dimensions),
            include_thinking=not args.no_thinking,
            progress=(lambda r: print(f"    round {r}", end="\r", flush=True)) if args.verbose else None,
        )
        frame.to_csv(run_dir / "judge.csv")
        if result.get("postrun"):
            ladder = judge_mod.judge_ladder(result["postrun"], client)
            (run_dir / "ladder.json").write_text(json.dumps(ladder, indent=2))
        verbal = judge_mod.verbal_detection_round(frame)
        print(f"    verbal detection round: {verbal}")
        if args.hand_labels:
            judge_mod.export_hand_labels(
                frame, Path(args.hand_labels) / f"{run_dir.name}.csv",
                run_id=result["run_id"], sample=args.hand_label_sample,
            )
    return 0


def cmd_reprobe(args) -> int:
    """Re-ask the post-run probes on saved runs, in their original context."""
    from .runner import reprobe

    root = Path(args.runs)
    dirs = [p for p in sorted(root.iterdir()) if (p / "run.json").exists()]
    if args.only:
        dirs = [p for p in dirs if args.only in p.name]
    for i, run_dir in enumerate(dirs, start=1):
        result = scoring.load_run(run_dir)
        model = args.model or f"{result['model']['backend']}:{result['model']['model']}"
        print(f"[{i}/{len(dirs)}] {run_dir.name}  ({model})", flush=True)
        client = make_client(model, **_client_kwargs(args))
        try:
            post = reprobe(result, client)
        except Exception as exc:
            print(f"    !! {type(exc).__name__}: {exc}", flush=True)
            continue
        (run_dir / "run.json").write_text(json.dumps(result, indent=2, default=str))
        with (run_dir / "transcript.jsonl").open("w") as fh:
            for item in result["transcript"]:
                fh.write(json.dumps(item, default=str) + "\n")
        (run_dir / "ladder.json").unlink(missing_ok=True)  # must be re-judged
        rungs = " | ".join(r["rung"] + ": " + r["answer"].strip().replace("\n", " ")[:60]
                           for r in post["ladder"])
        print(f"    {rungs}", flush=True)
    return 0


def cmd_plot(args) -> int:
    """Render runs as terminal text -- readable over SSH with no image viewer."""
    from . import termplot

    summary, frames = scoring.score_directory(args.runs)
    if summary.empty:
        print("no runs found")
        return 1
    rows = summary.to_dict("records")
    if args.only:
        rows = [r for r in rows if args.only in r["run_id"]]
    if args.survival or len(rows) > 1:
        termplot.survival(summary, channel="behavioural")
    for row in rows[: args.limit]:
        termplot.run_report(frames[row["run_id"]], row, compact=args.compact)
    return 0


def cmd_judge_check(args) -> int:
    """Check a judge model on notes with known labels before trusting it."""
    client = make_client(
        args.judge_model,
        temperature=args.judge_temperature,
        think=args.judge_think,
        num_predict=args.judge_num_predict,
    )
    frame = judge_mod.calibrate(client)
    cols = ["case"] + [c for c in frame.columns if c != "case"]
    print(frame[cols].to_string(index=False))
    summary = judge_mod.calibration_summary(frame)
    print("\n" + "  ".join(f"{k}={v:.2f}" for k, v in summary.items()))
    bad = [k for k, v in summary.items() if v < 0.8]
    if bad:
        print(f"\nWARNING: {', '.join(bad)} below 0.8 — do not run the grid with this judge")
        return 1
    return 0


def cmd_analyse(args) -> int:
    summary, frames = scoring.score_directory(args.runs)
    if summary.empty:
        print("no runs found")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    for run_id, df in frames.items():
        df.to_csv(out / f"rounds_{run_id}.csv", index=False)

    tables = analysis.report(summary, out)
    cols = [
        "run_id", "model", "rule", "alpha", "framing", "seed",
        "oracle_detection_round", "behavioural_round", "verbal_round",
        "first_probe_round", "final_z", "mean_z_last5", "ordering",
        "model_mae", "mae_D1", "mae_A",
    ]
    present = [c for c in cols if c in summary.columns]
    print("\n=== per-run summary ===")
    print(summary[present].to_string(index=False))
    for name, table in tables.items():
        if table.empty:
            continue
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))

    if args.plots:
        _plots(summary, frames, out)
        print(f"\nplots written to {out}")
    print(f"\ntables written to {out}")
    return 0


def _plots(summary: pd.DataFrame, frames: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    analysis.plot_z_trajectories(frames, ax=ax)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "z_trajectories.png", dpi=150)
    plt.close(fig)

    if summary["behavioural_round"].notna().any() or len(summary) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        analysis.plot_survival(summary, ax=ax)
        fig.tight_layout()
        fig.savefig(out / "survival_behavioural.png", dpi=150)
        plt.close(fig)

    for run_id, df in list(frames.items())[:8]:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        analysis.plot_paths(df, ax=axes[0])
        axes[0].set_title(run_id, fontsize=8)
        analysis.plot_oracle(df, ax=axes[1])
        fig.tight_layout()
        fig.savefig(out / f"run_{run_id}.png", dpi=150)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="perfcast", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_model_args(sp, default_model="echo"):
        sp.add_argument("--model", default=default_model, help="ollama:NAME | anthropic:NAME | echo")
        sp.add_argument("--temperature", type=float, default=0.7)
        sp.add_argument("--model-seed", type=int, default=None, help="ollama sampling seed")
        sp.add_argument("--num-ctx", type=int, default=16384)
        sp.add_argument("--num-predict", type=int, default=768,
                        help="raise well above 2000 when thinking is on")
        sp.add_argument("--think", dest="think", action="store_true", default=None)
        sp.add_argument("--no-think", dest="think", action="store_false")
        sp.add_argument("--shadow", action="store_true", help="collect shadow forecasts")
        sp.add_argument("--shadow-model", default=None)
        sp.add_argument("--shadow-every", type=int, default=1)
        sp.add_argument("--shadow-n", type=int, default=1,
                        help="draws per shadow forecast; >1 measures its own dispersion")
        sp.add_argument("--no-postrun", action="store_true")
        sp.add_argument("--out", default=str(DEFAULT_RUNS))
        sp.add_argument("--tag", default="")
        sp.add_argument("-v", "--verbose", action="store_true")

    sp = sub.add_parser("models", help="list ollama models")
    sp.add_argument("--host", default="http://localhost:11434")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("selftest", help="simulator, paths and oracle sanity checks")
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("run", help="one run")
    add_model_args(sp)
    sp.add_argument("--rule", type=int, default=1)
    sp.add_argument("--alpha", type=float, default=0.5)
    sp.add_argument("--framing", default="market", choices=["market", "river", "abstract"])
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--warmup", type=int, default=20)
    sp.add_argument("--horizon", type=int, default=40)
    sp.add_argument("--hidden-shock", action="store_true")
    sp.add_argument("--alpha-switch-round", type=int, default=None)
    sp.add_argument("--alpha-after", type=float, default=None)
    sp.add_argument("--level-shift-round", type=int, default=None)
    sp.add_argument("--level-shift-size", type=float, default=0.0)
    sp.add_argument("--hide-own-forecasts", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("pilot", help="2 seeds, rule 1, alpha 0.5, market framing")
    add_model_args(sp)
    sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    sp.add_argument("--horizon", type=int, default=40)
    sp.set_defaults(func=cmd_pilot)

    sp = sub.add_parser("grid", help="first-pass experiment grid")
    add_model_args(sp)
    sp.add_argument("--rules", type=int, nargs="+", default=[1, 2])
    sp.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.3, 0.7])
    sp.add_argument("--framings", nargs="+", default=["market", "abstract"])
    sp.add_argument("--seeds", type=int, default=10)
    sp.add_argument("--warmup", type=int, default=20)
    sp.add_argument("--horizon", type=int, default=40)
    sp.add_argument("--controls", action="store_true", help="add hidden-shock twins")
    sp.add_argument("--interventions", action="store_true", help="add the alpha-switch arm")
    sp.add_argument("--alpha-switch-round", type=int, default=30)
    sp.add_argument("--alpha-after", type=float, default=0.2)
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_grid)

    sp = sub.add_parser("sweep", help="the same conditions across several models")
    add_model_args(sp)
    sp.add_argument("--models", nargs="+", required=True)
    sp.add_argument("--rules", type=int, nargs="+", default=[1])
    sp.add_argument("--alphas", type=float, nargs="+", default=[0.5])
    sp.add_argument("--framings", nargs="+", default=["market"])
    sp.add_argument("--seeds", type=int, nargs="+", default=[0])
    sp.add_argument("--warmup", type=int, default=20)
    sp.add_argument("--horizon", type=int, default=40)
    sp.add_argument("--controls", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("reprobe", help="re-ask the post-run probes on saved runs")
    add_model_args(sp)
    sp.set_defaults(model=None)
    sp.add_argument("--runs", default=str(DEFAULT_RUNS))
    sp.add_argument("--only", default=None)
    sp.set_defaults(func=cmd_reprobe)

    sp = sub.add_parser("judge", help="label the verbal channel")
    sp.add_argument("--runs", default=str(DEFAULT_RUNS))
    sp.add_argument("--judge-model", default="echo")
    sp.add_argument("--judge-temperature", type=float, default=0.0)
    sp.add_argument("--judge-think", dest="judge_think", action="store_true", default=False)
    sp.add_argument("--no-judge-think", dest="judge_think", action="store_false")
    sp.add_argument("--judge-num-predict", type=int, default=512)
    sp.add_argument("--dimensions", nargs="+",
                    default=["verbal_level", "is_probe", "attribution", "evidence_cited"])
    sp.add_argument("--only", default=None, help="substring filter on run id")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--no-thinking", action="store_true",
                    help="judge only the published reasoning, not the chain of thought")
    sp.add_argument("--hand-labels", default=None, help="directory for hand-label CSVs")
    sp.add_argument("--hand-label-sample", type=int, default=10)
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_judge)

    sp = sub.add_parser("plot", help="render runs as terminal text (works over ssh)")
    sp.add_argument("--runs", default=str(DEFAULT_RUNS))
    sp.add_argument("--only", default=None, help="substring filter on run id")
    sp.add_argument("--limit", type=int, default=4)
    sp.add_argument("--compact", action="store_true", help="paths and oracle only")
    sp.add_argument("--survival", action="store_true")
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("judge-check", help="calibrate a judge model on known notes")
    sp.add_argument("--judge-model", default="echo")
    sp.add_argument("--judge-temperature", type=float, default=0.0)
    sp.add_argument("--judge-think", dest="judge_think", action="store_true", default=False)
    sp.add_argument("--no-judge-think", dest="judge_think", action="store_false")
    sp.add_argument("--judge-num-predict", type=int, default=512)
    sp.set_defaults(func=cmd_judge_check)

    sp = sub.add_parser("analyse", help="score runs and write the tables")
    sp.add_argument("--runs", default=str(DEFAULT_RUNS))
    sp.add_argument("--out", default="analysis")
    sp.add_argument("--plots", action="store_true")
    sp.set_defaults(func=cmd_analyse)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

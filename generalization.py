"""Leave-one-position-out: does an intervention reach an edit position it never trained on?

Onion and spiral compute ``scale(position)`` in closed form, so either returns a scale at a
position no training example used. For each held-out position p the grid trains on
source-modified edits at every other position -- train *and* val, so early stopping never
sees p -- and scores the full test set, which still edits at every position. That run's
accuracy at p is the transfer. The same config trained on every position (``[]`` in the
grid) is the in-distribution reference, and

    gap(p) = acc(trained without p, at p) - acc(trained on every position, at p)

is the number to read: 0 means nothing was lost by never seeing p. Raw held-out accuracy is
not, because onion already falls off at late positions when it does train on them. A
held-out run and its reference share a seed, hence a word split and a test set, so the gap
is paired per seed.

A per-position lookup (``low_rank-*``, ``per_pos``, DAS) has no closed form to carry over:
its row for p never receives a gradient and keeps its random initialisation, so it is not
in the grid.

    python -m intervention.generalization --n_jobs 6     # train what is missing, then print
    python -m intervention.generalization --models spiral_rope --seeds 42 43 44 --n_jobs 5
    python -m intervention.generalization --show         # print what is on disk, train nothing
    python -m intervention.generalization --show --plot plots/generalization/loo.pdf

From a notebook::

    from intervention.generalization import ensure_loo, loo_frame, render, plot_loo
    ensure_loo(); df = loo_frame(); render(df); plot_loo(df)
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from intervention.config import ExperimentConfig
from intervention.experiments.grid import expand_grid, load_grid_from_json, run_grid
from intervention.plotting.style import INK, method_color, method_label, paper_style

ROOT = Path(__file__).resolve().parent
GRID_PATH = ROOT / "generalization_grid.json"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results" / "generalization"

REFERENCE = -1       # ``held_out`` of the run trained on every position
MISSING = "--"       # what a cell shows when its runs have not finished
METRICS = ("acc_reference", "acc_heldout", "gap")
METRIC_LABELS = {"acc_reference": "trained on all", "acc_heldout": "held out", "gap": "gap"}
MARKERS = {"onion": "o", "spiral_rope": "D"}  # so a model is never told apart by colour alone


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
def loo_grid(models: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
             grid_path: Path = GRID_PATH) -> dict[str, list]:
    """The grid, optionally narrowed to a subset of its models and/or other CV seeds --
    e.g. one model first, the other later -- without editing the file."""
    grid = load_grid_from_json(grid_path)
    if models:
        unknown = sorted(set(models) - set(grid["model"]))
        if unknown:
            raise ValueError(f"models {unknown} are not in {grid_path.name}: {grid['model']}")
        grid["model"] = [m for m in grid["model"] if m in set(models)]
    if seeds:
        grid["seeds"] = [int(s) for s in seeds]
    return grid


def loo_configs(models: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                ) -> list[ExperimentConfig]:
    """The grid's configs, exactly as ``run_grid`` trains them. Each holds out at most one
    position: that is what lets a run's accuracy at its own held-out position be read off
    as *the* transfer to that position."""
    configs = expand_grid(loo_grid(models, seeds))
    multi = [c.run_name() for c in configs if len(c.data.heldout_positions) > 1]
    if multi:
        raise ValueError(f"leave-one-out holds out one position per run; got {multi}")
    return configs


def ensure_loo(
    results_dir: Path = RESULTS_DIR,
    cache_dir: Path = CACHE_DIR,
    n_jobs: int = 1,
    force: bool = False,
    verbose: bool = False,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> Path:
    """Train the grid (or its ``models`` x ``seeds`` subset) into its own results tree,
    skipping configs that already have a ``cv_summary.json`` -- whatever seeds that
    summary covers, so widening ``seeds`` later needs ``force``."""
    results_dir = Path(results_dir)
    configs = loo_configs(models, seeds)
    todo = [c for c in configs
            if force or not (results_dir / c.run_name() / "cv_summary.json").exists()]
    if not todo:
        print(f"[loo] up to date: {len(configs)} configs under {results_dir}")
        return results_dir
    print(f"[loo] {len(todo)} of {len(configs)} configs to train -> {results_dir}")
    run_grid(loo_grid(models, seeds), results_dir, cache_dir=Path(cache_dir), n_jobs=n_jobs,
             skip_existing=not force, verbose=verbose)
    return results_dir


# --------------------------------------------------------------------------- #
# Reading back
# --------------------------------------------------------------------------- #
def position_accuracy(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Long format, one row per (model, held_out, seed, test position): sequence accuracy
    and test-example count. ``held_out`` is the position the run never trained on, or
    :data:`REFERENCE` for the run trained on every position.

    Reads every finished seed on disk, whichever subset of models and seeds was trained,
    so a partial grid (one model so far, fewer seeds) reads back as just that."""
    cols = ["model", "held_out", "seed", "position", "n", "accuracy"]
    frames: list[pd.DataFrame] = []
    for cfg in loo_configs():
        held = cfg.data.heldout_positions[0] if cfg.data.heldout_positions else REFERENCE
        for path in sorted((Path(results_dir) / cfg.run_name()).glob("seed_*/predictions.csv")):
            seed = int(path.parent.name.removeprefix("seed_"))
            acc = (pd.read_csv(path).groupby("position")["match"]
                   .agg(accuracy="mean", n="size").reset_index())
            frames.append(acc.assign(model=cfg.method.model, held_out=held, seed=seed))
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def loo_frame(results_dir: Path = RESULTS_DIR, accuracy: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (model, held-out position p, seed): accuracy at p from the run that never
    saw p, from the reference run with the same seed, and their difference ``gap``.
    ``acc_reference`` (and so ``gap``) is NaN where that seed's reference is unfinished."""
    acc = position_accuracy(results_dir) if accuracy is None else accuracy
    keys = ["model", "seed", "position"]
    ref = (acc[acc["held_out"] == REFERENCE]
           .rename(columns={"accuracy": "acc_reference"})[[*keys, "acc_reference"]])
    held = (acc[acc["held_out"] == acc["position"]]
            .rename(columns={"accuracy": "acc_heldout"})[[*keys, "n", "acc_heldout"]])
    df = held.merge(ref, on=keys, how="left")
    df["gap"] = df["acc_heldout"] - df["acc_reference"]
    return df.sort_values(["position", "model", "seed"]).reset_index(drop=True)


def _ordered_models(models) -> list[str]:
    """Onion, then spiral, then anything else the grid grew, so tables and figures agree."""
    present = set(models)
    first = [m for m in ("onion", "spiral_rope") if m in present]
    return first + sorted(present - set(first))


def _cell(values: pd.Series, decimals: int, signed: bool) -> str:
    values = values.dropna()
    if values.empty:
        return MISSING
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    sign = "+" if signed else ""
    return f"{values.mean():{sign}.{decimals}f} ± {std:.{decimals}f}"


def render(frame: pd.DataFrame | None = None, decimals: int = 3,
           results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Held-out position (rows) x model x {trained on all, held out, gap}, as ``mean ± std``
    over seeds. The ``mean`` row averages each seed over its positions first, then takes
    mean ± std over seeds. ``n`` is the test examples at that position (mean over seeds)."""
    df = loo_frame(results_dir) if frame is None else frame
    if df.empty:
        return pd.DataFrame()
    overall = df.groupby(["model", "seed"])[list(METRICS)].mean().reset_index()
    rows = pd.concat([df.assign(position=df["position"].astype(str)),
                      overall.assign(position="mean")], ignore_index=True)
    order = [str(p) for p in sorted(df["position"].unique())] + ["mean"]

    n = df.groupby("position")["n"].mean()
    cols = {("", "n"): pd.Series({str(p): f"{v:.0f}" for p, v in n.items()} | {"mean": ""})}
    for model in _ordered_models(df["model"]):
        sub = rows[rows["model"] == model].groupby("position")
        for metric in METRICS:
            cols[(method_label(model), METRIC_LABELS[metric])] = sub[metric].apply(
                lambda s, d=decimals, signed=(metric == "gap"): _cell(s, d, signed))
    table = pd.DataFrame(cols).reindex(order)
    table.index.name = "held-out position"
    return table.fillna(MISSING)


def missing_runs(results_dir: Path = RESULTS_DIR, models: Sequence[str] | None = None,
                 seeds: Sequence[int] | None = None) -> list[str]:
    """``run_name`` of every config (of the ``models`` x ``seeds`` subset) with at least
    one seed not finished."""
    return [c.run_name() for c in loo_configs(models, seeds)
            if not all((Path(results_dir) / c.run_name() / f"seed_{s}" / "predictions.csv").exists()
                       for s in c.train.seeds)]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _mean_sd(df: pd.DataFrame, value: str) -> pd.DataFrame:
    g = df.groupby("position")[value]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1).fillna(0.0)})


def plot_loo(frame: pd.DataFrame | None = None, accuracy: pd.DataFrame | None = None,
             results_dir: Path = RESULTS_DIR, path: Path | None = None) -> Figure:
    """Left: accuracy at each edit position, trained on every position (line, band = ±1 sd
    over seeds) vs. with that position held out (markers, ±1 sd). Right: the paired gap.
    A model whose closed form carries over keeps its markers on its line and its gap at 0."""
    acc = position_accuracy(results_dir) if accuracy is None else accuracy
    df = loo_frame(accuracy=acc) if frame is None else frame
    models = _ordered_models(df["model"])
    held_positions = sorted(df["position"].unique())
    dodge = dict(zip(models, np.linspace(-0.12, 0.12, len(models)) if len(models) > 1 else [0.0]))

    fig, (ax_acc, ax_gap) = plt.subplots(1, 2, figsize=(8.0, 3.2), sharex=True)
    for model in models:
        color, marker, label = method_color(model), MARKERS.get(model, "s"), method_label(model)
        ref = acc[(acc["model"] == model) & (acc["held_out"] == REFERENCE)
                  & acc["position"].isin(held_positions)]
        if not ref.empty:
            r = _mean_sd(ref, "accuracy")
            ax_acc.plot(r.index, r["mean"], color=color, lw=1.8, label=f"{label}, trained on all")
            ax_acc.fill_between(r.index, r["mean"] - r["sd"], r["mean"] + r["sd"],
                                color=color, alpha=0.15, lw=0)
        sub = df[df["model"] == model]
        for ax, value, name in ((ax_acc, "acc_heldout", f"{label}, position held out"),
                                (ax_gap, "gap", label)):
            s = _mean_sd(sub.dropna(subset=[value]), value)
            if s.empty:
                continue
            ax.errorbar(s.index + dodge[model], s["mean"], yerr=s["sd"], fmt=marker, ms=6,
                        color=color, mec="white", mew=1.0, elinewidth=1.2, capsize=0,
                        label=name)

    ax_acc.set(ylim=(0, 1), ylabel="Sequence accuracy at that position",
               title="Accuracy at the edit position")
    ax_gap.axhline(0, color=INK, lw=0.8)
    ax_gap.set(ylabel="Held out − trained on all", title="Cost of never training on it")
    for ax in (ax_acc, ax_gap):
        ax.set_xlabel("Edit position")
        ax.set_xticks(held_positions)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
        ax.set_axisbelow(True)
    # One legend for both panels, above them: the gap panel reuses the held-out markers.
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="parallel CPU workers over configs (datasets are built first)")
    parser.add_argument("--force", action="store_true", help="retrain even if present")
    parser.add_argument("--show", action="store_true", help="print from disk, train nothing")
    parser.add_argument("--plot", type=Path, default=None, help="also save the figure here")
    parser.add_argument("--models", nargs="+", default=None,
                        help="train only these of the grid's models (e.g. spiral_rope)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="override the grid's CV seeds")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.show:
        ensure_loo(args.results, args.cache, n_jobs=args.n_jobs, force=args.force,
                   verbose=args.verbose, models=args.models, seeds=args.seeds)
    acc = position_accuracy(args.results)
    df = loo_frame(accuracy=acc)
    print("\nLeave-one-position-out: accuracy at the held-out edit position")
    print(render(df).to_string() if not df.empty else "(no finished runs)")
    if not df.empty:
        df.to_csv(args.results / "loo_by_seed.csv", index=False)
        print(f"\nper-seed rows -> {args.results / 'loo_by_seed.csv'}")
    pending = missing_runs(args.results, args.models, args.seeds)
    if pending:
        print(f"[warn] {len(pending)} config(s) not finished: "
              f"{', '.join(pending[:3])}{' ...' if len(pending) > 3 else ''}")
    if args.plot is not None and not df.empty:
        paper_style()
        plot_loo(df, acc, path=args.plot)
        print(f"figure -> {args.plot}")


if __name__ == "__main__":
    main()

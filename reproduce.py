"""Rebuild every artifact ``plotting/paper_plots.ipynb`` reads.

The notebook plots; this module produces what it plots. Four independent steps, each of
which no-ops when its output is already on disk, so it is safe to call from the notebook
as a "make sure this exists" cell and safe to re-run after an interruption:

    states     states_ds/{train,test}_states_*         encoder states over the lexicon
    runs       results/paper/<run_name>/               the trained interventions (paper_grid.json)
    baseline   plots/random_baseline/profiles.csv      untrained-encoder control
    surprisal  plots/surprisal/regression.csv          predictability control

From the shell::

    python -m intervention.reproduce                 # everything that is missing
    python -m intervention.reproduce --steps runs    # just the training runs
    python -m intervention.reproduce --n_jobs 4      # parallelise configs over CPUs
    python -m intervention.reproduce --force         # ignore what is already there

From the notebook::

    from intervention.reproduce import ensure_all
    paper = ensure_all()          # -> PaperArtifacts, with the paths the figures read

Budget a few hours for a cold ``runs`` step: the intervention itself is tiny, but every
(dataset, seed) has to be generated once — each candidate edit is checked against the
frozen repeat model — before any training starts. That work is cached under ``cache/``,
so the second run of the same grid only trains.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from intervention.config import ExperimentConfig
from intervention.experiments.grid import grid_iter, load_grid_from_json, run_grid

ROOT = Path(__file__).resolve().parent
PAPER_GRID = ROOT / "paper_grid.json"
RESULTS_DIR = ROOT / "results" / "paper"
CACHE_DIR = ROOT / "cache"

STEPS = ("states", "runs", "baseline", "surprisal")


@dataclass(frozen=True)
class PaperArtifacts:
    """Where each step left its output. The notebook reads these, never literal paths —
    a run directory's name is derived from its config, so it moves when the config does."""

    results_dir: Path
    states: dict[str, Path]
    baseline: Path | None
    surprisal: Path | None

    @property
    def runs(self) -> dict[str, Path]:
        """``run_name -> directory`` for every config in the paper grid."""
        return {cfg.run_name(): self.results_dir / cfg.run_name() for cfg in paper_configs()}

    def missing_runs(self) -> list[str]:
        return [name for name, d in self.runs.items() if not (d / "cv_summary.json").exists()]


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
def paper_configs(grid_path: Path = PAPER_GRID) -> list[ExperimentConfig]:
    """Expand ``paper_grid.json`` into configs, exactly as ``run_grid`` does.

    Sharing the expansion is what keeps the notebook honest: it asks a config for its
    ``run_name`` instead of hard-coding a directory that silently goes stale the next time
    a default changes.
    """
    grid = load_grid_from_json(grid_path)
    seeds = grid.pop("seeds", None)  # the CV dimension, applied to every config
    rows = list(grid_iter(grid))
    if seeds is not None:
        for row in rows:
            row["seeds"] = seeds
    return [ExperimentConfig.from_flat(row) for row in rows]


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def ensure_states(force: bool = False) -> dict[str, Path]:
    from intervention.state_analysis.extract_states import ensure_states as _ensure

    return _ensure(force=force)


def ensure_runs(
    results_dir: Path = RESULTS_DIR,
    cache_dir: Path = CACHE_DIR,
    grid_path: Path = PAPER_GRID,
    n_jobs: int = 1,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """Train the paper grid into ``results_dir`` (skipping configs already finished)."""
    results_dir = Path(results_dir)
    configs = paper_configs(grid_path)
    todo = [c for c in configs
            if force or not (results_dir / c.run_name() / "cv_summary.json").exists()]
    if not todo:
        print(f"[runs] up to date: {len(configs)} configs under {results_dir}")
        return results_dir

    print(f"[runs] {len(todo)} of {len(configs)} configs to train -> {results_dir}")
    run_grid(grid_path, results_dir, cache_dir=Path(cache_dir), n_jobs=n_jobs,
             skip_existing=not force, verbose=verbose)
    return results_dir


def ensure_baseline(force: bool = False) -> Path:
    from intervention.state_analysis.random_baseline import main as _baseline

    return _baseline(force=force)


def ensure_surprisal(force: bool = False) -> Path:
    from intervention.state_analysis.surprisal import PLOTS_DIR, main as _surprisal

    path = PLOTS_DIR / "regression.csv"
    if path.exists() and not force:
        print(f"[surprisal] up to date: {path}")
        return path
    _surprisal()
    return path


def ensure_all(
    steps: tuple[str, ...] = STEPS,
    results_dir: Path = RESULTS_DIR,
    cache_dir: Path = CACHE_DIR,
    grid_path: Path = PAPER_GRID,
    n_jobs: int = 1,
    force: bool = False,
    verbose: bool = False,
) -> PaperArtifacts:
    """Run the requested steps and return the paths the notebook's figures read."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown steps {sorted(unknown)}; choose from {STEPS}")

    states = ensure_states(force) if "states" in steps else {}
    if "runs" in steps:
        ensure_runs(results_dir, cache_dir, grid_path, n_jobs=n_jobs,
                    force=force, verbose=verbose)
    baseline = ensure_baseline(force) if "baseline" in steps else None
    surprisal = ensure_surprisal(force) if "surprisal" in steps else None

    artifacts = PaperArtifacts(Path(results_dir), states, baseline, surprisal)
    still_missing = artifacts.missing_runs()
    if still_missing:
        print(f"[warn] {len(still_missing)} run(s) still missing: "
              f"{', '.join(still_missing[:3])}{' ...' if len(still_missing) > 3 else ''}")
    return artifacts


def summary_table(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """One row per finished config: method, embedding, mean +/- std test accuracy."""
    rows = []
    for cfg in paper_configs():
        path = Path(results_dir) / cfg.run_name() / "cv_metrics.csv"
        if not path.exists():
            continue
        metrics = pd.read_csv(path)
        rows.append({
            "model": cfg.method.model,
            "embedding": "learned" if cfg.method.train_embedding else "fixed",
            "n_seeds": len(metrics),
            "test_acc_mean": metrics["final_test_acc"].mean(),
            "test_acc_std": metrics["final_test_acc"].std(ddof=1),
            "run_name": cfg.run_name(),
        })
    columns = ["model", "embedding", "n_seeds", "test_acc_mean", "test_acc_std", "run_name"]
    if not rows:  # nothing trained yet -- an empty frame, not a KeyError on sort
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["model", "embedding"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--grid", type=Path, default=PAPER_GRID)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="parallel CPU workers over configs (datasets are built first)")
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    ensure_all(tuple(args.steps), args.results, args.cache, args.grid,
               n_jobs=args.n_jobs, force=args.force, verbose=args.verbose)

    table = summary_table(args.results)
    if not table.empty:
        print("\n=== paper runs ===")
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

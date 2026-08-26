"""Figures for the paper, and the diagnostics for a single training run.

Two layers, in this order:

**Reading results.** ``Run`` is one ``seed_<n>/`` directory and ``CVRun`` is the config
directory above it; both load lazily and cache, so a notebook can hold twenty of them and
only pay for the ones it plots. ``load_runs`` sweeps a results tree and hands back the
whole set keyed by run name. Nothing below ever opens a file itself.

**Drawing.** Every figure function takes data (a frame, an array, a set of ``CVRun``),
returns the ``Figure``, and saves only when given a ``path``. That is what makes them
usable from ``paper_plots.ipynb`` — the notebook wants the figure inline — and from the
command line, which wants a png::

    python -m intervention.plotting.plots results/paper    # diagnostics for every run

A note on ``scales``. For the spiral methods the rotation is part of the intervention, so
``params.npz["scales"]`` is the *rotated* per-position vector that was actually applied —
the effective scale. Do not rotate it again.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from intervention.paths import get_phoneme_features, get_phoneme_to_id, get_wfe_dataset
from intervention.plotting.style import (
    RANDOM_COLOR,
    TRAINED_COLOR,
    label_color,
    method_color,
    method_label,
    paper_style,
)
from intervention.state_analysis.geometry import MAX_POS

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

DEFAULT_FEATURES = ["position", "Lexicality", "Size", "Morphology", "type-change", "Condition"]

__all__ = [
    "Run", "CVRun", "load_runs", "merge_predictions", "accuracy_table", "paper_style",
    "norm_by_position", "similarity_histogram", "similarity_by_position",
    "accuracy_by_position", "accuracy_bars", "scale_pca", "angle_by_position",
    "scale_norm_by_position",
    "dimension_by_position", "random_baseline", "training_curves",
    "accuracy_by_feature", "embedding_pca", "report_run", "report_all",
]


# --------------------------------------------------------------------------- #
# Reading results
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _wfe() -> pd.DataFrame:
    return get_wfe_dataset()


@lru_cache(maxsize=1)
def _features() -> dict[str, dict[str, str]]:
    return get_phoneme_features()


def _phoneme_type(phoneme) -> str | None:
    entry = _features().get(phoneme)
    return entry.get("Type", "other") if isinstance(entry, dict) else None


def _as_numeric(series: pd.Series) -> pd.Series:
    """``match`` arrives as True/False, "True"/"False" or 1/0 depending on the writer."""
    if series.dtype == object:
        series = series.astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    return pd.to_numeric(series, errors="coerce")


def merge_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Join a run's ``predictions.csv`` to the WFE frame's word-level factors.

    The join key is the input sequence with ``<EOS>`` stripped, which is how the same word
    is written on both sides. Adds the articulatory type of the edited-out and edited-in
    phoneme (and their pairing, ``type-change``), so accuracy can be split by what kind of
    substitution was asked for.
    """
    predictions = predictions.copy()
    predictions["input_no_eos"] = (
        predictions["input"].astype(str).str.replace("<EOS>", "", regex=False).str.strip()
    )
    if "match" in predictions:
        predictions["match"] = _as_numeric(predictions["match"])
    for column in ("token_acc", "position"):
        if column in predictions:
            predictions[column] = pd.to_numeric(predictions[column], errors="coerce")

    wfe = _wfe().copy()
    wfe["input_no_eos"] = wfe["No_Stress"].apply(
        lambda seq: " ".join(seq).strip() if isinstance(seq, (list, tuple)) else str(seq).strip()
    )
    merged = predictions.merge(
        wfe.drop(columns=[c for c in ("Phonemes", "No_Stress") if c in wfe]),
        on="input_no_eos", how="left",
    )

    merged["old-ph-type"] = merged["old_phoneme"].map(_phoneme_type)
    merged["new-ph-type"] = merged["new_phoneme"].map(_phoneme_type)
    both = merged["old-ph-type"].notna() & merged["new-ph-type"].notna()
    merged["type-change"] = np.where(
        both, merged["old-ph-type"].astype(str) + "-" + merged["new-ph-type"].astype(str), None
    )

    keep = ["Word", "Condition", "input", "target", "prediction", "position", "old_phoneme",
            "new_phoneme", "seq_len", "match", "token_acc", "old-ph-type", "new-ph-type",
            "type-change", "Lexicality", "Size", "Morphology", "Frequency", "Length",
            "Zipf_Frequency", "Part of Speech"]
    return merged[[c for c in keep if c in merged]]


@dataclass
class Run:
    """One trained seed: the ``seed_<n>/`` directory written by ``experiments.runner``."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def seed(self) -> int | None:
        name = self.path.name
        return int(name.split("_")[-1]) if name.startswith("seed_") else None

    @cached_property
    def config(self) -> dict:
        return json.loads((self.path / "config.json").read_text())

    @cached_property
    def history(self) -> pd.DataFrame:
        history = pd.read_csv(self.path / "history.csv")
        return history.assign(epoch=range(len(history)))

    @cached_property
    def params(self) -> dict[str, np.ndarray]:
        return dict(np.load(self.path / "params.npz", allow_pickle=True))

    @cached_property
    def predictions(self) -> pd.DataFrame:
        """``predictions.csv`` joined to the word-level factors (see ``merge_predictions``)."""
        return merge_predictions(pd.read_csv(self.path / "predictions.csv"))

    @property
    def scales(self) -> np.ndarray:
        """Per-position scale vectors, ``(max_position, state_dim)``, rotation included."""
        scales = self.params["scales"]
        return scales.reshape(scales.shape[0], -1)

    def __repr__(self) -> str:  # keeps notebook output readable
        return f"Run({self.path.name})"


@dataclass
class CVRun:
    """One config, cross-validated: the directory holding ``seed_*/`` and ``cv_metrics.csv``."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @cached_property
    def runs(self) -> list[Run]:
        return [Run(p) for p in sorted(self.path.glob("seed_*")) if (p / "config.json").exists()]

    @cached_property
    def config(self) -> dict:
        summary = self.path / "cv_summary.json"
        return json.loads(summary.read_text()) if summary.exists() else self.runs[0].config

    @cached_property
    def metrics(self) -> pd.DataFrame:
        """One row per seed: test accuracy/loss, val accuracy, best epoch."""
        return pd.read_csv(self.path / "cv_metrics.csv")

    @cached_property
    def predictions(self) -> pd.DataFrame:
        """Every seed's merged predictions, stacked, with a ``seed`` column."""
        return pd.concat([r.predictions.assign(seed=r.seed) for r in self.runs],
                         ignore_index=True)

    @cached_property
    def scales_by_seed(self) -> np.ndarray:
        """``(n_seeds, max_position, state_dim)`` — the effective (rotated) scale vectors."""
        return np.stack([r.scales for r in self.runs])

    @property
    def scales(self) -> np.ndarray:
        """Scale vectors of the first seed, the one the parameter figures illustrate."""
        return self.runs[0].scales

    @property
    def model(self) -> str:
        return self.config.get("model") or self.config.get("scale_param", "?")

    @property
    def embedding(self) -> str:
        """Whether the identity embedding was trained alongside the scale, or held fixed."""
        return "learned" if self.config.get("train_embedding") else "fixed"

    @property
    def label(self) -> str:
        return method_label(self.model)

    @property
    def color(self) -> str:
        return method_color(self.model)

    def __repr__(self) -> str:
        return f"CVRun({self.label}, {self.embedding}, n_seeds={len(self.runs)})"


def load_runs(results_dir: Path, names: Sequence[str] | None = None) -> dict[str, CVRun]:
    """Every cross-validated run under ``results_dir``, keyed by directory name.

    ``names`` restricts (and orders) the result — pass the run names a config expands to,
    and a missing one raises here rather than surfacing as an empty figure later.
    """
    results_dir = Path(results_dir)
    found = {p.parent.name: CVRun(p.parent) for p in sorted(results_dir.rglob("cv_metrics.csv"))}
    if names is None:
        return found
    missing = [n for n in names if n not in found]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} run(s) not under {results_dir}: {missing}\n"
            "Train them with `python -m intervention.reproduce --steps runs`."
        )
    return {n: found[n] for n in names}


def accuracy_table(runs: Iterable[CVRun], by: str = "position") -> pd.DataFrame:
    """Long format ``method / embedding / seed / <by> / accuracy``, one row per group.

    ``by="position"`` gives the accuracy curve across edit positions; any other column of
    the merged predictions works too (``Lexicality``, ``type-change``, ...).
    """
    frames = []
    for run in runs:
        grouped = (run.predictions.groupby(["seed", by])["match"].mean()
                   .reset_index(name="accuracy"))
        frames.append(grouped.assign(method=run.label, embedding=run.embedding))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _finish(fig: Figure, path: Path | None = None, tight: bool = True) -> Figure:
    """Every figure funnels through here: same tight-layout, same save behaviour."""
    if tight:
        fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def _ordered(runs: Iterable[CVRun]) -> list[CVRun]:
    """Stable plotting order: by method, then learned before fixed."""
    return sorted(runs, key=lambda r: (r.label, r.embedding != "learned"))


# --------------------------------------------------------------------------- #
# Figure 1 — the geometry of the states
# --------------------------------------------------------------------------- #
def norm_by_position(norms: pd.DataFrame, ylabel: str = r"$\|\Delta z\|$",
                     path: Path | None = None) -> Figure:
    """Distribution of update magnitude at each position (``geometry.norm_table``)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=norms, x="position", y="norm", ax=ax,
                color=sns.color_palette("colorblind")[0], fliersize=2, linewidth=1.2)
    ax.set_xlabel("Position", fontsize=20, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=8)
    ax.tick_params(axis="both", labelsize=15)
    sns.despine(ax=ax)
    return _finish(fig, path)


def similarity_histogram(similarities: dict[str, np.ndarray],
                         path: Path | None = None) -> Figure:
    """Overlaid cosine-similarity distributions, one per phoneme-category pair.

    The medians are drawn and labelled: whether C-V sits below C-C and V-V is the whole
    question, and the eye reads that off the dashed lines faster than off the humps.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("colorblind")
    for (label, values), color in zip(similarities.items(), palette):
        median = float(np.median(values))
        sns.histplot(values, label=label, alpha=0.4, stat="count", bins=50, ax=ax,
                     color=color, edgecolor="none")
        ax.axvline(median, linestyle="--", linewidth=1.8, color=color)
        ax.text(median, ax.get_ylim()[1] * 0.98, f"{median:.3f}", color=color,
                ha="center", va="bottom", fontsize=13, fontweight="bold", rotation=45)

    ax.set_xlabel("Cosine similarity", fontsize=20, labelpad=8)
    ax.set_ylabel("Count", fontsize=20, labelpad=8)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=16, frameon=False)
    sns.despine(ax=ax)
    return _finish(fig, path)


def similarity_by_position(similarity: pd.DataFrame, path: Path | None = None) -> Figure:
    """Same-phoneme vs different-phoneme similarity across position.

    Takes ``geometry.similarity_by_position``; the band is +/-1 std *across phonemes*, so
    it shows how consistent the effect is over the inventory, not how many tokens there are.
    """
    palette = sns.color_palette("colorblind")
    styles = {"same": (palette[0], "o", "Same phoneme"),
              "different": (palette[3], "^", "Different phoneme")}

    fig, ax = plt.subplots(figsize=(9, 4))
    for kind, (color, marker, label) in styles.items():
        part = similarity[similarity["kind"] == kind]
        ax.plot(part["position"], part["mean"], color=color, marker=marker,
                markersize=7, linewidth=2, label=label)
        ax.fill_between(part["position"], part["mean"] - part["std"],
                        part["mean"] + part["std"], color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel("Position", fontsize=14, labelpad=8)
    ax.set_ylabel("Mean cosine similarity", fontsize=14, labelpad=8)
    ax.set_xticks(sorted(similarity["position"].unique()))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, frameon=False)
    sns.despine(ax=ax)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Figure 2 — the intervention
# --------------------------------------------------------------------------- #
def accuracy_by_position(runs: Iterable[CVRun], max_pos: int = MAX_POS,
                         path: Path | None = None) -> Figure:
    """Intervention accuracy against edit position, one line per method x embedding.

    Colour is the method, dash is whether the identity embedding was learned; the band is
    +/-1 sd over CV seeds. ``max_pos`` is inclusive.
    """
    runs = _ordered(runs)
    table = accuracy_table(runs, by="position")
    table = table[table["position"] <= max_pos]
    order = list(dict.fromkeys(r.label for r in runs))

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=table, x="position", y="accuracy", hue="method", style="embedding",
                 hue_order=order, style_order=["learned", "fixed"],
                 palette={r.label: r.color for r in runs},
                 estimator="mean", errorbar="sd", err_style="band",
                 err_kws={"alpha": 0.18}, linewidth=2, markers=False, ax=ax)

    ax.set(xlabel="Position", ylabel="Intervention accuracy")
    ax.grid(axis="y", alpha=0.18)
    ax.grid(axis="x", alpha=0.08)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, title="")
    sns.despine(ax=ax)
    return _finish(fig, path)


def accuracy_bars(runs: Iterable[CVRun], path: Path | None = None) -> Figure:
    """Overall test accuracy per method, learned vs fixed identity embedding.

    Hatching (not a second colour) carries the embedding condition, so the method colours
    stay the same ones used in every other panel.
    """
    runs = _ordered(runs)
    table = pd.concat(
        [r.metrics.assign(method=r.label, embedding=r.embedding) for r in runs],
        ignore_index=True,
    )
    order = list(dict.fromkeys(r.label for r in runs))
    hue_order = [e for e in ("learned", "fixed") if e in set(table["embedding"])]

    fig, ax = plt.subplots(figsize=(1.9 * len(order) + 1.5, 4))
    sns.barplot(data=table, x="method", y="final_test_acc", hue="embedding",
                order=order, hue_order=hue_order, estimator="mean", errorbar="sd",
                capsize=0.12, width=0.72, edgecolor="white", linewidth=0.8, ax=ax)

    # One container per hue level, bars within it in `order` — no geometric guessing.
    colors = {r.label: r.color for r in runs}
    for container, embedding in zip(ax.containers, hue_order):
        for bar, method in zip(container, order):
            bar.set_facecolor(colors[method])
            bar.set_edgecolor("white")
            bar.set_alpha(0.7 if embedding == "learned" else 0.6)
            if embedding != "learned":
                bar.set_hatch("//")

    # Label above the error bar, not above the bar: the caps would otherwise run through
    # the text whenever the spread across seeds is wide.
    stats = table.groupby(["method", "embedding"])["final_test_acc"].agg(["mean", "std"])
    for container, embedding in zip(ax.containers, hue_order):
        for bar, method in zip(container, order):
            mean, std = stats.loc[(method, embedding)]
            std = 0.0 if pd.isna(std) else std
            ax.annotate(f"{mean:.2f} ± {std:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + std),
                        xytext=(0, 6), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9, weight="bold", clip_on=False)

    headroom = (table.groupby(["method", "embedding"])["final_test_acc"]
                .agg(lambda s: s.mean() + (s.std(ddof=1) if len(s) > 1 else 0)).max())
    ax.set(xlabel="", ylabel="Intervention accuracy", ylim=(0, min(1.05, headroom + 0.15)))
    ax.grid(axis="y", alpha=0.18)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(title="Identity embedding", frameon=True, loc="upper left", fontsize=9,
              title_fontsize=10, handles=[
                  Patch(facecolor="white", edgecolor="black", label=e,
                        hatch=None if e == "learned" else "//") for e in hue_order])
    sns.despine(ax=ax)
    return _finish(fig, path)


def scale_pca(scales: np.ndarray, max_pos: int = MAX_POS, polar: bool = False,
              title: str = "", path: Path | None = None) -> Figure:
    """The learned per-position scale vectors, projected to their top two components.

    Position is the colour, so the onion prediction (points strung along one ray, only the
    radius changing) and the spiral prediction (the angle turning with position) are
    visibly different pictures. ``polar=True`` replots the same projection as
    (angle, radius), which makes a constant per-step rotation read as even spacing.

    ``scales`` must be the *effective* vectors — ``Run.scales`` / ``params.npz["scales"]``
    already include the spiral rotation.
    """
    scales = np.asarray(scales)
    scales = scales.reshape(scales.shape[0], -1)[: max_pos + 1]
    pca = PCA(n_components=2)
    projected = pca.fit_transform(scales)
    position = np.arange(len(projected))
    var1, var2 = pca.explained_variance_ratio_[:2] * 100

    if polar:
        fig, ax = plt.subplots(figsize=(6.2, 5.6), subplot_kw={"projection": "polar"})
        scatter = ax.scatter(np.arctan2(projected[:, 1], projected[:, 0]),
                             np.linalg.norm(projected, axis=1), c=position, cmap="viridis",
                             s=70, alpha=0.95, edgecolors="white", linewidths=0.7)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(135)
        ax.grid(alpha=0.22, linewidth=0.8)
        ax.spines["polar"].set_visible(False)
        ax.text(0.02, 0.02, f"PC1: {var1:.1f}%\nPC2: {var2:.1f}%", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.85", alpha=0.9))
    else:
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        scatter = ax.scatter(projected[:, 0], projected[:, 1], c=position,
                             cmap="viridis", s=60, edgecolors="white", linewidths=0.6)
        ax.set(xlabel=f"PC1 ({var1:.1f}% var)", ylabel=f"PC2 ({var2:.1f}% var)")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    bar = fig.colorbar(scatter, ax=ax, pad=0.1 if polar else 0.02, shrink=0.82)
    bar.set_label("Position")
    bar.set_ticks([0, len(projected) - 1])
    if title:
        ax.set_title(title, pad=12)
    return _finish(fig, path)


def angle_by_position(angles: dict[str, np.ndarray], path: Path | None = None) -> Figure:
    """Angle to the position-0 scale vector, per method.

    Flat at zero is the onion hypothesis — position rescales one fixed direction. A curve
    that climbs is direction turning with position, which is the spiral claim.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4))
    markers = ["o", "s", "^", "D", "v"]
    for (label, values), marker in zip(angles.items(), markers):
        ax.plot(np.arange(len(values)), values, marker=marker, markersize=5,
                linewidth=1.8, label=label, color=label_color(label))
    ax.set(xlabel="Position", ylabel="Angle to position 0 (degrees)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    return _finish(fig, path)


def scale_norm_by_position(runs: Iterable[CVRun], max_pos: int = MAX_POS,
                           log: bool = True, path: Path | None = None) -> Figure:
    """Magnitude of the learned scale vector by position, mean and 95% CI over CV seeds.

    The companion to ``angle_by_position``: a spiral is a radius and an angle, and this is
    the radius. Log scale by default — an unbounded parameterisation can run three orders
    of magnitude past a bounded one, which a linear axis renders as one curve and a flat
    line along zero.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for run in _ordered(runs):
        norms = np.linalg.norm(run.scales_by_seed[:, : max_pos + 1], axis=-1)  # (seeds, positions)
        mean = norms.mean(axis=0)
        n = len(norms)
        ci = 1.96 * norms.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        style = "-" if run.embedding == "learned" else "--"
        position = np.arange(len(mean))
        ax.plot(position, mean, style, color=run.color, linewidth=2,
                label=f"{run.label} ({run.embedding})")
        lower = np.maximum(mean - ci, mean * 1e-3) if log else mean - ci
        ax.fill_between(position, lower, mean + ci, color=run.color, alpha=0.18, linewidth=0)

    if log:
        ax.set_yscale("log")
    ax.set(xlabel="Position", ylabel="Norm of scale vector")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Appendix
# --------------------------------------------------------------------------- #
def dimension_by_position(dimensions: pd.DataFrame, metric: str = "d90",
                          path: Path | None = None) -> Figure:
    """``geometry.dimension_by_position``: how many directions each position's cloud uses."""
    labels = {"d90": "PCs for 90% of variance", "participation_ratio": "Participation ratio"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for (embed_type, part), marker in zip(dimensions.groupby("embed_type", sort=False),
                                          ["o", "s", "^", "D"]):
        ax.plot(part["position"], part[metric], marker=marker, markersize=5,
                linewidth=1.8, label=embed_type)
    ax.set(xlabel="Position", ylabel=labels.get(metric, metric))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    return _finish(fig, path)


def random_baseline(profiles: pd.DataFrame, target: str = "delta_state", max_pos: int = MAX_POS,
                    rescale: bool = False, path: Path | None = None) -> Figure:
    """Trained encoder against randomly initialised ones (``random_baseline.profiles``).

    ``rescale=True`` divides each curve by its own position-1 value, which is the honest
    comparison: random weights land on an arbitrary scale, so only the shape transfers.
    """
    part = profiles[(profiles["target"] == target)
                    & profiles["position"].between(1, max_pos)]

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in part.groupby("model", sort=False):
        y = group.set_index("position")["norm"]
        if rescale:
            y = y / y.iloc[0]
        trained = label == "trained"
        ax.plot(y.index, y.values, label=label,
                color=TRAINED_COLOR if trained else RANDOM_COLOR,
                linewidth=2.8 if trained else 1.3,
                marker="o" if trained else None, markersize=7,
                zorder=3 if trained else 1)
    if rescale:
        ax.axhline(1.0, color="k", linewidth=0.6, linestyle=":")

    ax.set(xlabel="Position",
           ylabel=r"$\|\Delta z\|$ relative to position 1" if rescale else r"$\|\Delta z\|$")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Per-run diagnostics
# --------------------------------------------------------------------------- #
def training_curves(history: pd.DataFrame, title: str = "",
                    path: Path | None = None) -> Figure:
    """Loss and accuracy per epoch for every split present in ``history.csv``."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for ax, metric, label in zip(axes, ("loss", "acc"), ("Loss", "Accuracy")):
        columns = [f"{s}_{metric}" for s in ("train", "val", "test")
                   if f"{s}_{metric}" in history]
        long = history.melt(id_vars="epoch", value_vars=columns,
                            var_name="split", value_name=metric)
        long["split"] = long["split"].str.replace(f"_{metric}", "", regex=False)
        sns.lineplot(data=long, x="epoch", y=metric, hue="split", marker="o", ax=ax)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.3)
        ax.legend(title="Split", frameon=False)

    final = history.iloc[-1]
    summary = "\n".join(f"{s.capitalize()}: {final[f'{s}_acc']:.4f}"
                        for s in ("train", "val", "test") if f"{s}_acc" in history)
    axes[1].text(0.95, 0.05, summary, transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                       alpha=0.8, edgecolor="gray"))
    if title:
        fig.suptitle(title, fontsize=11)
    return _finish(fig, path)


def accuracy_by_feature(predictions: pd.DataFrame, features: Sequence[str] = DEFAULT_FEATURES,
                        title: str = "", path: Path | None = None) -> Figure | None:
    """Mean accuracy within each level of every requested factor, one panel per factor."""
    features = [f for f in features if f in predictions]
    if not features:
        return None

    columns = min(3, len(features))
    rows = (len(features) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4 * rows), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        summary = (predictions.groupby(feature)["match"].mean().reset_index()
                   .sort_values("match", ascending=False))
        sns.barplot(data=summary, x=feature, y="match", color="steelblue", ax=ax)
        ax.set(xlabel="", ylabel="Accuracy", title=feature, ylim=(0, 1))
        if summary[feature].dtype == object:
            ax.tick_params(axis="x", rotation=45)
    for extra in axes.flat[len(features):]:
        fig.delaxes(extra)

    fig.suptitle(title or "Accuracy by feature", fontsize=12)
    return _finish(fig, path)


def embedding_pca(embedding: np.ndarray, title: str = "",
                  path: Path | None = None) -> Figure:
    """The learned identity embedding in two components, labelled and coloured by type."""
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embedding)
    id_to_phoneme = {i: p for p, i in get_phoneme_to_id().items()}
    table = pd.DataFrame({
        "PC1": projected[:, 0], "PC2": projected[:, 1],
        "phoneme": [id_to_phoneme.get(i, f"id{i}") for i in range(len(projected))],
    })
    table["type"] = table["phoneme"].map(_phoneme_type).fillna("other")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=table, x="PC1", y="PC2", hue="type", palette="Set2",
                    s=90, edgecolor="k", ax=ax)
    for _, row in table.iterrows():
        ax.text(row["PC1"], row["PC2"], row["phoneme"], fontsize=7, alpha=0.8)
    variance = pca.explained_variance_ratio_ * 100
    ax.set(title=title or "Identity embedding (PCA)",
           xlabel=f"PC1 ({variance[0]:.1f}% var)", ylabel=f"PC2 ({variance[1]:.1f}% var)")
    ax.legend(title="Phoneme type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    return _finish(fig, path)


def report_run(run_dir: Path, features: Sequence[str] = DEFAULT_FEATURES) -> None:
    """Write the diagnostic figures for one trained seed, next to its artifacts."""
    run = Run(run_dir)
    config = run.config
    title = (f"{config.get('model')} | state={config.get('state_mode')} | "
             f"init={config.get('embedding_init')} | train_embed={config.get('train_embedding')}")

    plt.close(training_curves(run.history, title=title, path=run.path / "training_curves.png"))
    if "scales" in run.params:
        plt.close(scale_pca(run.scales, title=title, path=run.path / "scale_pca.png"))
        plt.close(scale_pca(run.scales, polar=True, path=run.path / "scale_pca_polar.png"))
    if "embedding" in run.params:
        plt.close(embedding_pca(run.params["embedding"], title=title,
                                path=run.path / "embedding_pca.png"))

    predictions = run.predictions
    predictions.to_csv(run.path / "merged_predictions.csv", index=False)
    figure = accuracy_by_feature(predictions, features, title=f"Accuracy by feature — {title}",
                                 path=run.path / "accuracy_by_features.png")
    if figure is not None:
        plt.close(figure)


def report_all(results_dir: Path, features: Sequence[str] = DEFAULT_FEATURES) -> None:
    """Diagnostics for every trained seed under ``results_dir``; one bad run never stops the batch."""
    run_dirs = sorted({p.parent for p in Path(results_dir).rglob("config.json")})
    if not run_dirs:
        print(f"No runs (config.json) found under {results_dir}")
        return
    for run_dir in run_dirs:
        print(f"Plotting {run_dir} ...")
        try:
            report_run(run_dir, features)
        except Exception as exc:
            print(f"  skipped ({exc})")


if __name__ == "__main__":
    import sys

    paper_style()
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if (target / "config.json").exists():
        report_run(target)
    else:
        report_all(target)

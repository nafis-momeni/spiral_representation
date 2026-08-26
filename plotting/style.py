"""Shared look for the report figures: one matplotlib style, one palette.

Imported by ``state_analysis.surprisal``, ``plotting.plots`` and the plotting notebooks,
so every panel agrees on colour, line weight and spacing instead of each module inventing
its own. A method keeps its colour across every figure it appears in, which is what lets
the reader carry "purple is the spiral" from one panel to the next.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

INK = "#2B2B2B"
COLORS = {"All": "#334155", "Consonant": "#1565C0", "Vowel": "#E07B39", "EOS": "#9E9E9E"}
SURPRISAL_COLOR = "#7B5AA6"
STRATA_COLORS = ["#9CC7E8", "#3E80D1", "#123B66"]  # low -> high surprisal

# Trained vs randomly initialised encoder (the untrained control).
TRAINED_COLOR, RANDOM_COLOR = "#1565C0", "#9E9E9E"

# The intervention families, keyed by ``MethodConfig.model``. Unknown models fall back to
# the seaborn "colorblind" cycle rather than colliding with a named one.
METHOD_LABELS = {
    "onion": "onion",
    "spiral_rope": "spiral",
    "low_rank-8": "low rank (r=8)",
    "low_rank-4": "low rank (r=4)",
    "das": "DAS",
    "das_auto": "DAS (learned vars)",
}
METHOD_COLORS = {
    "onion": "#C16288",
    "spiral_rope": "#926DDC",
    "low_rank-8": "#4EADB4",
    "low_rank-4": "#77B27A",
    "das": "#B08A3E",
    "das_auto": "#7A7A7A",
}
_FALLBACK = sns.color_palette("colorblind")


def method_label(model: str) -> str:
    """Display name for a ``model`` id ('spiral_rope' -> 'spiral')."""
    return METHOD_LABELS.get(model, model)


def method_color(model: str) -> str:
    """Stable colour for a ``model`` id, by name or (for unknown ones) by hash."""
    if model in METHOD_COLORS:
        return METHOD_COLORS[model]
    return _FALLBACK[hash(model) % len(_FALLBACK)]


def method_palette(models) -> dict[str, str]:
    """``{display label: colour}`` for a set of models, ready to hand to seaborn."""
    return {method_label(m): method_color(m) for m in models}


_LABEL_TO_MODEL = {label: model for model, label in METHOD_LABELS.items()}


def label_color(label: str) -> str:
    """The colour for a *display* label, so a figure keyed by label still matches the rest."""
    return method_color(_LABEL_TO_MODEL.get(label, label))


def paper_style():
    """Figure defaults: thin dark spines, muted grid, labels with room to breathe."""
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
    })

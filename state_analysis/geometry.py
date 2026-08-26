"""What the encoder states look like, as numbers.

The descriptive half of the paper asks three questions of the same state dump, and each
gets one function here. They return tidy frames; ``plotting.plots`` draws them, and
nothing in this module touches matplotlib.

    norm_table            how big is the update at each position?          -> ||Dz|| by position
    category_similarity   do consonants and vowels move in the same way?   -> cosine sims by C/V pair
    similarity_by_position is the direction of the update phoneme-specific? -> cosine sims by position
    dimension_by_position  how many directions does it use?                -> d90 / participation ratio
    angles_to_first        does the intervention's scale rotate with position?

``delta[0]`` is the state itself rather than a difference (position 0 has no
predecessor), so every position-indexed analysis here can drop it — see ``drop_first``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from intervention.paths import get_phoneme_features

# Last position any figure shows, inclusive -- the same bound ``surprisal`` and
# ``random_baseline`` use. Past here the per-position n is in the double digits.
MAX_POS = 12


def phoneme_categories() -> tuple[list[str], list[str]]:
    """``(vowels, consonants)`` as the shipped feature table labels them."""
    features = get_phoneme_features()
    vowels = [p for p, f in features.items() if f.get("Type") == "V"]
    consonants = [p for p, f in features.items() if f.get("Type") == "C"]
    return vowels, consonants


# --------------------------------------------------------------------------- #
# Magnitude
# --------------------------------------------------------------------------- #
def norm_table(
    dataset,
    embed_type: str = "delta_state",
    max_pos: int = MAX_POS,
    drop_first: bool = True,
) -> pd.DataFrame:
    """One row per token: ``position`` and the L2 norm of its state (or delta).

    ``max_pos`` is inclusive, here and everywhere else in this module."""
    table = dataset.metadata.copy()
    table["norm"] = np.linalg.norm(dataset.get_embeddings(embed_type), axis=-1)
    if drop_first:
        table = table[table["position"] != 0]
    return table[table["position"] <= max_pos].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Direction: identity
# --------------------------------------------------------------------------- #
def _subsample(array: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(array) <= n:
        return array
    return array[rng.choice(len(array), n, replace=False)]


def category_similarity(
    dataset,
    category_a: list[str],
    category_b: list[str],
    embed_type: str = "delta_state",
    max_samples: int = 1_000_000,
    seed: int = 42,
) -> np.ndarray:
    """Cosine similarities between the states of two phoneme categories.

    Both sides are subsampled *before* the pairwise matrix is formed — the full lexicon
    would be a 200k x 200k matrix. When the two categories are the same set, only the
    strict upper triangle counts, so a token is never compared with itself.
    """
    rng = np.random.default_rng(seed)
    states = dataset.get_embeddings(embed_type)
    side = int(np.sqrt(max_samples)) + 1
    a = _subsample(states[dataset.metadata["phoneme"].isin(category_a).to_numpy()], side, rng)
    b = _subsample(states[dataset.metadata["phoneme"].isin(category_b).to_numpy()], side, rng)

    matrix = cosine_similarity(a, b)
    same = set(category_a) == set(category_b)
    sims = matrix[np.triu_indices_from(matrix, k=1)] if same else matrix.ravel()
    return _subsample(sims, max_samples, rng)


def similarity_by_position(
    dataset,
    embed_type: str = "delta_state",
    min_pos: int = 1,
    max_pos: int = MAX_POS,
    max_per_cell: int = 1000,
    max_per_position: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Mean cosine similarity at each position, within vs across phoneme identity.

    Each phoneme contributes one number per position (the mean similarity of its tokens
    to tokens of the same phoneme, and to tokens of every other phoneme, all at that
    position); the returned ``mean``/``std`` summarise *across phonemes*, so a frequent
    phoneme cannot dominate the curve.

    Position 0 is excluded by default: ``delta[0]`` is the state itself, which is a
    deterministic function of the phoneme with no context, so same-phoneme similarity
    there is exactly 1 by construction and says nothing about the encoder.

    Returns long format: ``position, kind ("same"|"different"), mean, std, n_phonemes``.
    """
    rng = np.random.default_rng(seed)
    states = dataset.get_embeddings(embed_type)
    metadata = dataset.metadata

    # Indices per (phoneme, position), capped so one cell can't blow up the matrix.
    cells = {
        key: _subsample(group.index.to_numpy(), max_per_cell, rng)
        for key, group in metadata.groupby(["phoneme", "position"])
    }
    phonemes = metadata["phoneme"].unique()

    rows = []
    for position in range(min_pos, max_pos + 1):
        same = [
            _mean_offdiagonal(cosine_similarity(states[idx]))
            for phoneme in phonemes
            if len(idx := cells.get((phoneme, position), np.empty(0, int))) > 1
        ]
        rows.append(_summarize(position, "same", same))
        rows.append(_summarize(position, "different",
                               _cross_phoneme_means(states, cells, phonemes, position,
                                                    max_per_position, rng)))
    return pd.DataFrame(rows)


def _mean_offdiagonal(matrix: np.ndarray) -> float:
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return float(upper.mean()) if upper.size else np.nan


def _cross_phoneme_means(states, cells, phonemes, position, cap, rng) -> list[float]:
    """For each phoneme: its mean similarity to tokens of *other* phonemes, same position."""
    index, labels = [], []
    for phoneme in phonemes:
        idx = cells.get((phoneme, position), np.empty(0, int))
        index.append(idx)
        labels.append(np.full(len(idx), phoneme))
    if not index:
        return []
    index, labels = np.concatenate(index), np.concatenate(labels)
    if len(index) < 2:
        return []
    if len(index) > cap:
        keep = rng.choice(len(index), cap, replace=False)
        index, labels = index[keep], labels[keep]

    matrix = cosine_similarity(states[index])
    means = []
    for phoneme in phonemes:
        rows, cols = labels == phoneme, labels != phoneme
        if rows.any() and cols.any():
            means.append(float(matrix[np.ix_(rows, cols)].mean()))
    return means


def _summarize(position: int, kind: str, values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "position": position,
        "kind": kind,
        "mean": float(np.nanmean(array)) if array.size else np.nan,
        "std": float(np.nanstd(array)) if array.size else np.nan,
        "n_phonemes": int(np.sum(~np.isnan(array))) if array.size else 0,
    }


# --------------------------------------------------------------------------- #
# Direction: how many of them
# --------------------------------------------------------------------------- #
def intrinsic_dimension(states: np.ndarray) -> tuple[int, float]:
    """``(d90, participation_ratio)`` for one cloud of states.

    ``d90`` is the hard count — how many principal components carry 90% of the variance.
    The participation ratio is the soft version, ``(sum var)^2 / sum(var^2)``: it does not
    depend on a threshold, and it falls when variance concentrates in a few directions.
    """
    centred = states - states.mean(axis=0)
    variance = np.linalg.svd(centred, compute_uv=False) ** 2 / (len(states) - 1)
    d90 = int(np.searchsorted(np.cumsum(variance) / variance.sum(), 0.90) + 1)
    return d90, float(variance.sum() ** 2 / (variance**2).sum())


def dimension_by_position(
    dataset,
    embed_types: tuple[str, ...] = ("state", "delta_state"),
    max_pos: int = MAX_POS,
) -> pd.DataFrame:
    """``d90`` and participation ratio per position, for each requested state type."""
    positions = dataset.metadata["position"].to_numpy()
    rows = []
    for embed_type in embed_types:
        states = dataset.get_embeddings(embed_type)
        for position in range(max_pos + 1):
            mask = positions == position
            if mask.sum() <= 1:
                continue
            d90, ratio = intrinsic_dimension(states[mask])
            rows.append({"embed_type": embed_type, "position": position,
                         "d90": d90, "participation_ratio": ratio, "n": int(mask.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The learned intervention
# --------------------------------------------------------------------------- #
def angles_to_first(scales: np.ndarray, max_pos: int = MAX_POS) -> np.ndarray:
    """Angle in degrees between ``scales[0]`` and each later position's scale vector.

    Truncated at ``max_pos`` (inclusive) like everything else here, so a caller never
    has to slice and get the off-by-one wrong.
    """
    scales = scales[: max_pos + 1]
    reference = scales[0]
    cosines = scales @ reference / (np.linalg.norm(scales, axis=1) * np.linalg.norm(reference))
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))

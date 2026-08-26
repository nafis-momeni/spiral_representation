"""Dump the frozen repeat model's encoder states for a word list.

Every descriptive analysis in the paper (norms by position, delta-state similarity,
intrinsic dimension, the surprisal control) reads one of these dumps rather than
re-running the encoder, so they all see exactly the same states.

    python -m intervention.state_analysis.extract_states            # train + test
    python -m intervention.state_analysis.extract_states --sets all # + the SSP frame
    python -m intervention.state_analysis.extract_states --force    # rebuild anyway

Each set writes ``states_ds/<name>_embeddings.npz`` + ``states_ds/<name>_metadata.csv``,
loaded back with ``StatesDataset.load(str(STATES_DIR / name))``. Already-written sets are
skipped, so this is safe to call from a notebook as a "make sure it exists" step.
"""
from __future__ import annotations

import argparse
import time
from ast import literal_eval
from pathlib import Path

import pandas as pd
import torch

from intervention.models.repeat_model import get_model
from intervention.paths import (
    DATASETS_DIR,
    STATES_DIR,
    get_phoneme_features,
    get_phoneme_to_id,
    get_train_dataset,
    resolve_weights,
)
from intervention.state_analysis.states_extract import StateExtractor, StatesDataset
from intervention.utils import seed_everything, set_device

MODEL_NAME = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
WEIGHTS = "resources/weights/1024_75.pth"

_CONVERTERS = {"Word": str, "Phonemes": literal_eval, "No_Stress": literal_eval}

# name -> (frame loader, column holding the label for each row). ``ssp`` is a synthetic
# sonority frame with no orthographic word, so its condition label stands in for one.
SETS: dict[str, tuple] = {
    "train_states": (get_train_dataset, "Word"),
    "test_states": (lambda: pd.read_csv(DATASETS_DIR / "wfe.csv", converters=_CONVERTERS), "Word"),
    "ssp_states": (lambda: pd.read_csv(DATASETS_DIR / "ssp.csv", converters=_CONVERTERS), "Type"),
}
DEFAULT_SETS = ("train_states", "test_states")


def load_repeat_model(device: torch.device) -> torch.nn.Module:
    model = get_model(MODEL_NAME)
    model.load_state_dict(
        torch.load(resolve_weights(WEIGHTS), map_location=device, weights_only=True)
    )
    return model.to(device).eval()


def extract(name: str, extractor: StateExtractor, out_dir: Path = STATES_DIR) -> Path:
    """Extract one named set and write it under ``out_dir``; returns the path stem."""
    load_frame, word_col = SETS[name]
    frame = load_frame()
    features = get_phoneme_features()

    start = time.perf_counter()
    dataset = StatesDataset.from_dataframe(frame, extractor, word_col=word_col)
    # C / V / EOS, so downstream plots can split by phoneme type without re-reading the
    # feature table. Anything outside the table (i.e. <EOS>) falls in the EOS bucket.
    dataset.metadata["phoneme_type"] = (
        dataset.metadata["phoneme"].map(lambda p: features.get(p, {}).get("Type", "EOS"))
    )

    stem = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save(str(stem))
    print(f"  {name}: {len(frame):,} sequences -> {len(dataset):,} tokens "
          f"in {time.perf_counter() - start:.1f}s")
    return stem


def ensure_states(
    names: tuple[str, ...] = DEFAULT_SETS,
    out_dir: Path = STATES_DIR,
    force: bool = False,
    device: torch.device | None = None,
) -> dict[str, Path]:
    """Extract the named sets unless they are already on disk. Returns name -> path stem."""
    out_dir = Path(out_dir)
    missing = [
        n for n in names
        if force or not (out_dir / f"{n}_embeddings.npz").exists()
    ]
    stems = {n: out_dir / n for n in names}
    if not missing:
        print(f"[states] up to date: {', '.join(names)}")
        return stems

    seed_everything(42)
    device = device or set_device()
    extractor = StateExtractor(load_repeat_model(device), get_phoneme_to_id(), device)
    print(f"[states] extracting {', '.join(missing)} -> {out_dir}")
    for name in missing:
        extract(name, extractor, out_dir)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS),
                        choices=[*SETS, "all"], help="which word lists to extract")
    parser.add_argument("--out", type=Path, default=STATES_DIR)
    parser.add_argument("--force", action="store_true", help="re-extract even if present")
    args = parser.parse_args()

    names = tuple(SETS) if "all" in args.sets else tuple(args.sets)
    ensure_states(names, out_dir=args.out, force=args.force)


if __name__ == "__main__":
    main()

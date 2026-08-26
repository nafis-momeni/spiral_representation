import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import xarray as xr
import torch
from ast import literal_eval
from typing import List, Dict, Tuple, Optional


class StateExtractor:
    """Extract LSTM encoder hidden and cell states"""

    def __init__(self, model, phoneme_to_id:dict, device):
        self. model = model
        self.phoneme_to_id = phoneme_to_id
        self.device = device
        self.model.eval()

    def _to_input_ids(self, phonemes: list[str]) -> torch.Tensor:
        """converts phonemes list to input ids tensor [1, seq_len]"""
        ids = [self.phoneme_to_id[p] for p in phonemes]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
    def _get_final_hidden(self, input_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """gets the final output of encoder (h, c) as numpy arrays of shape (hidden size)"""
        with torch.no_grad():
            h, c = self.model.encoder(input_ids) # [num_layers, batch, hidden_size]
            return h[-1, 0].cpu().numpy(), c[-1,0].cpu().numpy()
    
    def extract_sequential(self, phonemes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Feed phonemes one-by-one (prefix 1..N), return states at each step. 
        return: h_states, c_states: (seq_len, hidden_size)
        """
        h_list, c_list = [], []
        for i in range(1, len(phonemes)+1):
            input_ids = self._to_input_ids(phonemes[:i])
            h,c = self._get_final_hidden(input_ids)
            h_list.append(h)
            c_list.append(c)
        return np.stack(h_list), np.stack(c_list)
    
    def extract_final(self, phonemes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """get the final states of the full sequence"""
        input_ids = self._to_input_ids(phonemes)
        h,c = self._get_final_hidden(input_ids)
        return h, c

    @torch.no_grad()
    def extract_batch(
        self, sequences: list[list[str]], batch_size: int = 512
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """``extract_sequential`` for many sequences at once, unrolled instead of re-fed.

        An LSTM started from a zero state reaches the same (h, c) after ``k`` steps
        whether the prefix is re-encoded from scratch or the state is carried, so this
        returns exactly what ``extract_sequential`` does — but in O(L) steps per sequence
        over a padded batch rather than O(L^2) one word at a time, which is the difference
        between seconds and minutes over the 30k-word lexicon.
        """
        pad_id = self.phoneme_to_id["<PAD>"]
        encoder = self.model.encoder
        out: list[tuple[np.ndarray, np.ndarray]] = []

        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            lengths = [len(s) for s in batch]
            width = max(lengths)

            ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=self.device)
            for i, seq in enumerate(batch):
                ids[i, : len(seq)] = torch.tensor(
                    [self.phoneme_to_id[p] for p in seq], dtype=torch.long, device=self.device
                )

            embedded = encoder.embedding(ids)
            state, hs, cs = None, [], []
            for t in range(width):
                _, state = encoder.recurrent(embedded[:, t : t + 1], state)
                hs.append(state[0][-1])
                cs.append(state[1][-1])
            h = torch.stack(hs, dim=1).cpu().numpy()   # (B, width, hidden)
            c = torch.stack(cs, dim=1).cpu().numpy()
            out.extend((h[i, :n], c[i, :n]) for i, n in enumerate(lengths))
        return out


@dataclass
class StatesDataset:
    """Stores extracted states for an entire dataset.
    
    Keeps metadata (scalars/strings) in a DataFrame,
    and high-dimensional embeddings in separate numpy arrays.
    Rows are aligned by index.
    
    Attributes:
        metadata: DataFrame with columns [seq_id, wo, position, phoneme]
        h: np.ndarray of shape (N, hidden_size) — hidden states
        c: np.ndarray of shape (N, hidden_size) — cell states
        delta_h: np.ndarray of shape (N, hidden_size) — h deltas (first = h itself)
        delta_c: np.ndarray of shape (N, hidden_size) — c deltas (first = c itself)
    """
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    h: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    delta_h: Optional[np.ndarray] = None
    delta_c: Optional[np.ndarray] = None
    state: Optional[np.ndarray] = None
    delta_state: Optional[np.ndarray] = None

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        extractor: StateExtractor,
        phoneme_col: str = "No_Stress",
        word_col: str = "Word",
        append_eos: bool = True,
        batch_size: int = 512,
    ) -> "StatesDataset":
        """Build a dataset from a frame of phoneme sequences.

        ``state``/``delta_state`` are filled in as ``[h, c]`` concatenated, so a caller
        never has to remember to stitch them together. ``delta[0]`` is the state itself,
        not a difference — position 0 has no predecessor (analyses drop it for that
        reason; see ``state_analysis.surprisal``).
        """
        sequences = [
            list(row[phoneme_col]) + (["<EOS>"] if append_eos else [])
            for _, row in df.iterrows()
        ]
        per_word = extractor.extract_batch(sequences, batch_size=batch_size)

        meta_rows = [
            {"seq_id": seq_id, "word": word, "position": pos, "phoneme": phoneme}
            for seq_id, word, phonemes in zip(df.index, df[word_col], sequences)
            for pos, phoneme in enumerate(phonemes)
        ]

        def with_deltas(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            deltas = [np.diff(s, axis=0, prepend=s[0:1]) for s in seqs]
            for delta, s in zip(deltas, seqs):
                delta[0] = s[0]
            return np.concatenate(seqs), np.concatenate(deltas)

        h, delta_h = with_deltas([h for h, _ in per_word])
        c, delta_c = with_deltas([c for _, c in per_word])
        return StatesDataset(
            metadata=pd.DataFrame(meta_rows).reset_index(drop=True),
            h=h, c=c, delta_h=delta_h, delta_c=delta_c,
            state=np.concatenate([h, c], axis=1),
            delta_state=np.concatenate([delta_h, delta_c], axis=1),
        )
    
    def __len__(self):
        return len(self.metadata)
    
    def get_mask(self, **filters) -> np.ndarray:
        """get a boolean mask for rows matching filters
        
        examples:
        ds.get_mask(phonemes="P")
        ds.get_mask(position=2, phoneme="AO")
        """
        mask = np.ones(len(self), dtype=bool)
        for col, val in filters.items():
            if isinstance(val,list):
                mask &= self.metadata[col].isin(val).values
            else:
                mask &= (self.metadata[col] ==val).values  
            
        return mask
    def get_embeddings(self, embed_type:str, mask:Optional[np.ndarray]=None)-> np.ndarray:
        """get embeddings by type optionally filtered"""
        arr = getattr(self, embed_type)
        if mask is not None:
           return arr[mask]
        return arr

    def copy(self) -> "StatesDataset":
        """Return a deep copy of metadata and embedding arrays."""
        return StatesDataset(
            metadata=self.metadata.copy(deep=True),
            h=self.h.copy() if self.h is not None else None,
            c=self.c.copy() if self.c is not None else None,
            delta_h=self.delta_h.copy() if self.delta_h is not None else None,
            delta_c=self.delta_c.copy() if self.delta_c is not None else None,
            state=self.state.copy() if self.state is not None else None,
            delta_state=self.delta_state.copy() if self.delta_state is not None else None,
        )

    def save(self, path: str):
        """save dataset to dist (npz + csv)"""
        arrays = {
            "h": self.h,
            "c": self.c,
            "delta_h": self.delta_h,
            "delta_c": self.delta_c,
        }
        if self.state is not None:
            arrays["state"] = self.state
        if self.delta_state is not None:
            arrays["delta_state"] = self.delta_state
        np.savez_compressed(f"{path}_embeddings.npz", **arrays)
        self.metadata.to_csv(f"{path}_metadata.csv", index=False)
    
    @staticmethod
    def load(path: str) -> "StatesDataset":
        "load dataset"
        data = np.load(f"{path}_embeddings.npz")
        metadata = pd.read_csv(f"{path}_metadata.csv")
        return StatesDataset(
            metadata=metadata,
            h=data["h"], c=data["c"],
            delta_h=data["delta_h"], delta_c=data["delta_c"],
            state=data["state"] if "state" in data.files else None,
            delta_state=data["delta_state"] if "delta_state" in data.files else None,
        )
    
def build_xarray_dataset(
    df: pd.DataFrame, 
    extractor: StateExtractor, 
    vowels: List[str], 
    consonants: List[str],
    word_col: str = "Word",
    phoneme_col: str = "No_Stress",
    append_eos: bool = True
) -> xr.Dataset:
    """Extracts states and builds a padded xarray Dataset."""
    words, all_h, all_c = [], [], []
    phonemes_padded, types_padded = [], []
    lengths = []
    
    # Determine determining max length
    max_len = df[phoneme_col].apply(len).max() + (1 if append_eos else 0)
    
    for _, row in df.iterrows():
        phonemes = list(row[phoneme_col])
        if append_eos:
            phonemes.append("<EOS>")
        
        length = len(phonemes)
        lengths.append(length)
        words.append(row[word_col])
        
        # Get states
        h_seq, c_seq = extractor.extract_sequential(phonemes)
        
        # Pad to max_len
        pad_len = max_len - length
        if pad_len > 0:
            h_seq = np.pad(h_seq, ((0, pad_len), (0, 0)), constant_values=np.nan)
            c_seq = np.pad(c_seq, ((0, pad_len), (0, 0)), constant_values=np.nan)
            phonemes.extend([""] * pad_len)
            
        all_h.append(h_seq)
        all_c.append(c_seq)
        
        # Phoneme types
        p_types = ["V" if p in vowels else ("C" if p in consonants else ("EOS" if p == "<EOS>" else "")) for p in phonemes]
        phonemes_padded.append(phonemes)
        types_padded.append(p_types)

    # Convert to arrays
    h = np.stack(all_h)  # (num_words, max_len, hidden_size)
    c = np.stack(all_c)
    
    # Deltas
    # dh[0] = h[0], effectively doing diff with padding
    dh = np.diff(h, axis=1, prepend=np.full_like(h[:, :1, :], np.nan))
    dc = np.diff(c, axis=1, prepend=np.full_like(c[:, :1, :], np.nan))
    # Fill actual first step with base value instead of nan
    for i, length in enumerate(lengths):
        if length > 0:
            dh[i, 0] = h[i, 0]
            dc[i, 0] = c[i, 0]

    # Combine states (h+c)
    state = np.concatenate([h, c], axis=2)
    delta_state = np.concatenate([dh, dc], axis=2)

    # Build dataset
    return xr.Dataset(
        data_vars={
            "h": (("word", "step", "hidden_dim"), h),
            "c": (("word", "step", "hidden_dim"), c),
            "delta_h": (("word", "step", "hidden_dim"), dh),
            "delta_c": (("word", "step", "hidden_dim"), dc),
            "state": (("word", "step", "state_dim"), state),
            "delta_state": (("word", "step", "state_dim"), delta_state),
        },
        coords={
            "word": words,
            "step": np.arange(max_len),
            "hidden_dim": np.arange(h.shape[-1]),
            "state_dim": np.arange(state.shape[-1]),
            "word_length": ("word", lengths),
            "phoneme": (("word", "step"), phonemes_padded),
            "phoneme_type": (("word", "step"), types_padded),
        }
    )
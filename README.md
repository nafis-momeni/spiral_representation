# Intervention experiments

Code for *Geometry in the Neural Processing of Phoneme Sequences: How Does Phoneme Identity
Bind to Position?*

This project tests the geometry of phoneme position in RNNs causally. It trains a small
intervention on the frozen phoneme repeat model, editing the encoder state to substitute one
phoneme, and measures whether the decoder then produces the edited word. We build on previous
work on position representation in RNNs. The onion hypothesis states that hidden-state
magnitude encodes position while state direction encodes identity. We test that hypothesis on
English words and find it incomplete. Position is encoded by magnitude together with a
systematic rotation of direction, a dynamic we formalise as a **spiral** model.


## Run

Requires Python 3.11 or newer. 

```bash
uv venv --python 3.11
source .venv/bin/activate   
uv pip install -e .
```

### Reproducing the paper

`plotting/paper_plots.ipynb` draws every figure in the paper and nothing else — the
artifacts it reads are produced by one command:

```bash
python -m intervention.reproduce --n_jobs 6    
```

That runs four independent steps, each skipped when its output already exists
(`--steps states runs baseline surprisal` selects a subset, `--force` rebuilds):


- `states`:  `states_ds/{train,test}_states_*` — encoder states over the lexicon 
- `runs` : `results/paper/<run_name>/` — the trained interventions in `paper_grid.json` 
- `baseline`: `plots/random_baseline/profiles.csv` — untrained-encoder control 
- `surprisal` : `plots/surprisal/regression.csv` — predictability control 



### Exploring

Training and plotting are separate steps. Every command below is run from the repo root.

```bash
python -m intervention.main --mode grid                     # grid_config.json, CV per config
python -m intervention.main --mode grid --n_jobs 4          # parallelise configs across CPU cores
python -m intervention.main --mode single --seeds 42 43 44  # one config, cross-validated
```

`grid_config.json` is the scratch grid for exploration; `paper_grid.json` is fixed and
is what the paper reports.


## Models

One field, `model`, selects the intervention. The **scale** family adds
`scale(position) · (emb(new) − emb(old))` to the state and differs only in how
`scale(position)` is parameterised. All models share the same data, trainer,
cross-validation, and plotting.

| `model` | what it does |
| --- | --- |
| `onion` | magnitude only, no rotation. `scale(p) = g · γ^p + p·β + b`, per dimension. This is the onion hypothesis. |
| `spiral_rope` | exponential-decay magnitude plus a RoPE-style rotation of the scale vector, with a learned frequency per plane of dimensions. This is the spiral hypothesis. |
| `low_rank-r` | unconstrained baseline. `scale(p)` is a per-position coefficient vector over `r` learned basis directions (default `r = 4`). |
| `das` | distributed alignment search. Rotate the state, swap the edited position's fixed block, rotate back. |
| `das_auto` | same, but every channel *learns* which variable it belongs to (or none), so variable sizes are learned rather than fixed. `params.npz` reports `var_sizes`, the channel count per variable. |

Also available as scale parameterisations: `per_pos`, `one_scale`, `linear`, `expo_decay`,
`expo_unbounded`, `plane_spiral`, `spiral_expo`, `spiral_lie`.

## Config

Grids are a flat `{key: [values]}` JSON, and every combination becomes one
`ExperimentConfig`. Key fields are `dataset` (real-real | source-modified |
modified-source), `model`, `state_mode`, `embedding_init`, `train_embedding`, and `seeds`
(the CV dimension, a list applied to every config rather than a swept axis). Unknown keys
raise.

## Resources

`resources/` holds every input the pipeline needs, so nothing is resolved against the
working directory or a parent repository. Reach it through `intervention.paths`, never by
building a relative path.

```
weights/      1024_75.pth                     the frozen repeat model checkpoint
phonemes/     phonemes_to_id_{sn,sw}.json     vocabulary, without / with stress
datasets/     training.csv                    real words the sequences are drawn from
              phonemes.csv                    per-phoneme articulatory features
              wfe[_with_repetition].csv       held-out word/non-word evaluation frame
embeddings/   phoneme_state_embeddings.npz    token-state statistics for `delta_*` inits
              ngram{2,3}_state_embeddings.npz same over n-grams
```

The shipped files *are* the input. A missing file raises instead of silently rebuilding
from the parent SWP project's grapheme-to-phoneme stack, which is not a dependency here.
`embeddings/*.npz` is regenerated with `python -m intervention.data.delta_embeddings`
(needs `states_ds/`).

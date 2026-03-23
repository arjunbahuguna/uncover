# cover_eval.py — Cover Song Detection Evaluation

A self-contained, model-agnostic evaluation module for **cover song identification** (also called *version detection*). Drop it into any project and get standard metrics with minimal boilerplate.

---

## Metrics

| Key in results dict | Full name | Range | Better |
|---|---|---|---|
| `mAP` | mean Average Precision | [0, 1] | ↑ higher |
| `MR1` | Mean Rank-1 | [1, N] | ↓ lower |
| `NAR` | Normalised Average Rank | [0, 100] | ↓ lower |
| `R@1`, `R@10`, `R@100` | Recall @ K | [0, 1] | ↑ higher |

### Definitions

**mAP** — For each query, compute the Average Precision over the ranked list of candidates, then average across all queries. The primary ranking quality metric.

**MR1** — Mean Rank-1: the average 1-based position of the *closest* true cover in the ranked list. A score of 1.0 means the model always returns a true cover at rank 1.

**NAR** — Normalised Average Rank (rank percentile). For each true cover in the ranked list, counts the fraction of *irrelevant* items that rank above it. Averaged over all true covers and queries, then scaled to [0, 100]. See [Korzeniowski et al.](https://publications.hevs.ch/index.php/publications/show/125).

**Recall@K** — Fraction of queries for which at least one true cover appears in the top-K results.

---

## Requirements

```
torch >= 1.12
```

No other dependencies.

---

## Installation

Just copy `cover_eval.py` into your project:

```bash
cp cover_eval.py your_project/
```

---

## Quick Start

### Option A — You have a model with a `.distances()` method

```python
import torch
from cover_eval import evaluate, print_results

# Your model must expose:
#   model.distances(q, c, qmask=None, cmask=None, redux_strategy=None)
#   returning a (Q, D) tensor of pairwise distances (lower = more similar)

results = evaluate(
    model,
    queries_z=query_embeddings,         # Tensor (Q, S, C)
    queries_c=query_clique_ids,         # Tensor (Q,)  — cover group ID
    queries_i=query_song_ids,           # Tensor (Q,)  — unique track ID
    candidates_z=database_embeddings,   # Tensor (D, S, C)
    candidates_c=database_clique_ids,   # Tensor (D,)
    candidates_i=database_song_ids,     # Tensor (D,)
    recall_ks=(1, 10, 100),             # which R@K to compute
    verbose=True,
)

print_results(results)
```

### Option B — You already have a distance matrix

```python
from cover_eval import evaluate_from_distances, print_results

# dist_matrix[q, d] = distance from query q to candidate d
results = evaluate_from_distances(
    dist_matrix,                        # Tensor (Q, D)
    queries_c=query_clique_ids,
    queries_i=query_song_ids,
    candidates_c=database_clique_ids,
    candidates_i=database_song_ids,
    recall_ks=(1, 10, 100),
)

print_results(results)
```

---

## What the results dict contains

```python
{
    # --- aggregate scalars (report these in papers) ---
    "mAP":   float,         # mean Average Precision
    "MR1":   float,         # Mean Rank-1
    "NAR":   float,         # mean Normalised Average Rank
    "R@1":   float,         # Recall@1  (fraction with hit in top-1)
    "R@10":  float,         # Recall@10
    "R@100": float,         # Recall@100

    # --- per-query tensors (useful for error analysis) ---
    "ap":    Tensor (Q,),   # per-query AP
    "r1":    Tensor (Q,),   # per-query rank-1
    "nar":   Tensor (Q,),   # per-query NAR
    "r@1":   Tensor (Q,),   # per-query recall@1
    "r@10":  Tensor (Q,),   # per-query recall@10
    "r@100": Tensor (Q,),   # per-query recall@100
}
```

---

## Common patterns

### Evaluating during training (validation loop)

```python
model.eval()
results = evaluate(model, val_qz, val_qc, val_qi, val_cz, val_cc, val_ci,
                   recall_ks=(1, 10))
print(f"Epoch {epoch} — mAP: {results['mAP']:.4f}  R@10: {results['R@10']:.4f}")
```

### Memory-limited GPUs

Use `batch_size_candidates` to chunk the distance computation:

```python
results = evaluate(
    model, ...,
    batch_size_candidates=512,   # process 512 candidates at a time
)
```

### Using the query set as its own database (standard benchmark setup)

Just pass the same tensors for both queries and candidates. The module automatically excludes self-matches using `queries_i` / `candidates_i`.

```python
results = evaluate(
    model,
    queries_z=embeddings,    candidates_z=embeddings,
    queries_c=clique_ids,    candidates_c=clique_ids,
    queries_i=song_ids,      candidates_i=song_ids,
)
```

### Error analysis on hard queries

```python
worst_queries_idx = results["ap"].argsort()[:10]   # 10 lowest AP
print("Worst queries by AP:", worst_queries_idx)
```

---

## Model interface contract

`cover_eval.evaluate()` calls:

```python
model.distances(q, c, qmask=None, cmask=None, redux_strategy=None)
```

| Argument | Shape | Description |
|---|---|---|
| `q` | `(1, S, C)` | Single query embedding (float) |
| `c` | `(D, S, C)` | All (or a batch of) candidate embeddings (float) |
| `qmask` | `(1, S)` or None | Boolean mask for query positions |
| `cmask` | `(D, S)` or None | Boolean mask for candidate positions |
| `redux_strategy` | any | Model-specific aggregation hint |
| **returns** | `(1, D)` | Pairwise distances (lower = more similar) |

If your model uses a different signature, use `evaluate_from_distances()` instead and compute distances externally.

---

## Running the built-in demo

```bash
python cover_eval.py
```

This runs a toy cosine-distance model on synthetic data and prints results for both `evaluate()` and `evaluate_from_distances()`.
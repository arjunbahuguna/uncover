"""
cover_eval.py
=============
Evaluation metrics for cover song identification / version detection.

Supported metrics
-----------------
- mAP   : mean Average Precision  (primary ranking quality metric)
- MR1   : Mean Rank-1             (is the top result a true cover?)
- NAR   : Normalised Average Rank (rank percentile, lower = better)
- R@K   : Recall at K             (fraction of queries whose top-K results contain ≥1 true cover)

All metrics work on pre-computed pairwise distance matrices or on raw
embeddings, making the module model-agnostic.

Typical usage
-------------
See the module-level docstring in README.md, or the quick example at the
bottom of this file under `if __name__ == "__main__"`.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Core per-query helpers
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _average_precision(distances: Tensor, is_match: Tensor) -> Tensor:
    """
    Compute Average Precision for a single query.

    Parameters
    ----------
    distances : Tensor, shape (N,)
        Distance from the query to each candidate (lower = more similar).
    is_match : Tensor, shape (N,), bool or float
        True / 1 for candidates that are genuine covers of the query.

    Returns
    -------
    Tensor scalar – AP in [0, 1].
    """
    assert distances.ndim == 1 and is_match.ndim == 1
    assert len(distances) == len(is_match)
    rel = is_match.float()
    assert rel.sum() >= 1, "Query has no positive candidates."

    order = torch.argsort(distances)
    rel = rel[order]
    rank = torch.arange(1, len(rel) + 1, device=distances.device, dtype=torch.float32)
    prec_at_k = torch.cumsum(rel, dim=0) / rank          # precision@k for each k
    ap = (prec_at_k * rel).sum() / rel.sum()
    return ap


@torch.inference_mode()
def _rank_of_first_correct(distances: Tensor, is_match: Tensor) -> Tensor:
    """
    Return the 1-based rank of the closest true cover for a single query.

    Parameters
    ----------
    distances : Tensor, shape (N,)
    is_match  : Tensor, shape (N,)

    Returns
    -------
    Tensor scalar – rank in {1, …, N}.
    """
    assert distances.ndim == 1 and is_match.ndim == 1
    assert len(distances) == len(is_match)
    rel = is_match.float()
    assert rel.sum() >= 1, "Query has no positive candidates."

    order = torch.argsort(distances)
    rel = rel[order]
    # argmax on a binary tensor returns the index of the first 1
    r1 = (torch.argmax(rel) + 1).float()
    return r1


@torch.inference_mode()
def _normalised_average_rank(distances: Tensor, is_match: Tensor) -> Tensor:
    """
    Compute Normalised Average Rank (rank percentile) for a single query.

    The metric counts, for each relevant item, what *fraction* of irrelevant
    items appear before it in the ranked list, averaged over all relevant items.
    A perfect score is 0.0; a score of 100.0 is worst possible.

    Reference: https://publications.hevs.ch/index.php/publications/show/125

    Parameters
    ----------
    distances : Tensor, shape (N,)
    is_match  : Tensor, shape (N,)

    Returns
    -------
    Tensor scalar – NAR in [0, 100].
    """
    assert distances.ndim == 1 and is_match.ndim == 1
    assert len(distances) == len(is_match)
    rel = is_match.float()
    assert rel.sum() >= 1, "Query has no positive candidates."

    order = torch.argsort(distances)
    rel = rel[order]
    n_irrelevant = rel.sum().item()   # will be reused below

    # Fraction of irrelevant items seen so far at each position
    # (unbiased version: perfect retrieval gives 0.0)
    cum_irrel = torch.cumsum(1.0 - rel, dim=0)
    n_irrel_total = (1.0 - rel).sum().clamp(min=1e-9)   # avoid div-by-zero
    norm_rank = cum_irrel / n_irrel_total                 # in [0, 1]

    nar = (rel * norm_rank).sum() / rel.sum()
    return 100.0 * nar


@torch.inference_mode()
def _recall_at_k(distances: Tensor, is_match: Tensor, k: int) -> Tensor:
    """
    Recall@K for a single query: 1 if ≥1 true cover is in the top-K results,
    else 0.

    Parameters
    ----------
    distances : Tensor, shape (N,)
    is_match  : Tensor, shape (N,)
    k         : int

    Returns
    -------
    Tensor scalar – 0.0 or 1.0.
    """
    assert distances.ndim == 1 and is_match.ndim == 1
    assert len(distances) == len(is_match)
    rel = is_match.bool()
    assert rel.sum() >= 1, "Query has no positive candidates."

    order = torch.argsort(distances)
    topk_match = rel[order[:k]]
    return topk_match.any().float()


# ---------------------------------------------------------------------------
# Batch evaluation – main public API
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate(
    model,
    queries_z: Tensor,          # (Q, S, C)  query embeddings
    queries_c: Tensor,          # (Q,)        query clique ids
    queries_i: Tensor,          # (Q,)        query song ids
    candidates_z: Tensor,       # (D, S, C)  database embeddings
    candidates_c: Tensor,       # (D,)        database clique ids
    candidates_i: Tensor,       # (D,)        database song ids
    queries_m: Tensor | None = None,        # optional query masks
    candidates_m: Tensor | None = None,     # optional candidate masks
    redux_strategy=None,
    recall_ks: tuple[int, ...] = (1, 10, 100),
    batch_size_candidates: int | None = None,
    verbose: bool = False,
) -> dict[str, float | Tensor]:
    """
    Run full evaluation for a cover-song retrieval model.

    Parameters
    ----------
    model
        Any object that exposes a ``distances(q, c, ...)`` method returning a
        (Q, D) tensor of pairwise distances (lower = more similar).
    queries_z : Tensor  (Q, S, C)
        Embeddings for all query tracks.
    queries_c : Tensor  (Q,)
        Clique (cover group) ID for each query.
    queries_i : Tensor  (Q,)
        Unique song ID for each query (used to exclude self-matches).
    candidates_z : Tensor  (D, S, C)
        Embeddings for the retrieval database.
    candidates_c : Tensor  (D,)
        Clique ID for each candidate.
    candidates_i : Tensor  (D,)
        Unique song ID for each candidate.
    queries_m : Tensor or None
        Boolean mask for query embeddings (True = valid position).
    candidates_m : Tensor or None
        Boolean mask for candidate embeddings.
    redux_strategy
        Passed through to ``model.distances``; controls how sequence
        positions are aggregated.
    recall_ks : tuple of int
        Which K values to evaluate Recall@K for.  Default: (1, 10, 100).
    batch_size_candidates : int or None
        If set, distances are computed in chunks to reduce peak GPU memory.
    verbose : bool
        Print progress every 100 queries.

    Returns
    -------
    dict with keys:
        'mAP'   – float, mean Average Precision
        'MR1'   – float, Mean Rank-1
        'NAR'   – float, mean Normalised Average Rank (lower = better)
        'R@{k}' – float for each k in recall_ks  (e.g. 'R@10')
        'ap'    – Tensor (Q,) per-query AP
        'r1'    – Tensor (Q,) per-query rank-1
        'nar'   – Tensor (Q,) per-query NAR
        'r@{k}' – Tensor (Q,) per-query recall@k (lowercase keys)
    """
    model.eval()

    Q = len(queries_i)
    ap_list  = []
    r1_list  = []
    nar_list = []
    rk_lists = {k: [] for k in recall_ks}

    for n in range(Q):
        if verbose and n % 100 == 0:
            print(f"  evaluating query {n}/{Q} …", flush=True)

        # ---- compute distances for query n ----
        dist = _compute_distances(
            model, n,
            queries_z, candidates_z,
            queries_m, candidates_m,
            redux_strategy, batch_size_candidates,
        )

        # ---- ground truth mask ----
        is_match = candidates_c == queries_c[n]   # same clique

        # ---- exclude the query track itself from candidates ----
        is_self = candidates_i == queries_i[n]
        dist     = torch.where(is_self, torch.full_like(dist, torch.inf), dist)
        is_match = torch.where(is_self, torch.zeros_like(is_match), is_match)

        # ---- per-query metrics ----
        ap_list.append(_average_precision(dist, is_match))
        r1_list.append(_rank_of_first_correct(dist, is_match))
        nar_list.append(_normalised_average_rank(dist, is_match))
        for k in recall_ks:
            rk_lists[k].append(_recall_at_k(dist, is_match, k))

    ap  = torch.stack(ap_list)
    r1  = torch.stack(r1_list)
    nar = torch.stack(nar_list)

    results: dict = {
        # --- aggregate scalars (the numbers you'll report in papers) ---
        "mAP":  ap.mean().item(),
        "MR1":  r1.mean().item(),
        "NAR":  nar.mean().item(),
        # --- per-query tensors (useful for error analysis) ---
        "ap":   ap,
        "r1":   r1,
        "nar":  nar,
    }

    for k in recall_ks:
        rk = torch.stack(rk_lists[k])
        results[f"R@{k}"] = rk.mean().item()   # e.g. "R@10"
        results[f"r@{k}"] = rk                 # e.g. "r@10" (per-query)

    return results


# ---------------------------------------------------------------------------
# Distance-only evaluation (when you already have a distance matrix)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_from_distances(
    dist_matrix: Tensor,        # (Q, D)
    queries_c: Tensor,          # (Q,)
    queries_i: Tensor,          # (Q,)
    candidates_c: Tensor,       # (D,)
    candidates_i: Tensor,       # (D,)
    recall_ks: tuple[int, ...] = (1, 10, 100),
    verbose: bool = False,
) -> dict[str, float | Tensor]:
    """
    Evaluate directly from a pre-computed distance matrix.

    This is convenient when distances are computed externally (e.g. via FAISS
    or a custom GPU kernel) and you just need the metrics.

    Parameters
    ----------
    dist_matrix : Tensor  (Q, D)
        Pairwise distances; ``dist_matrix[q, d]`` is the distance from query
        *q* to candidate *d*.
    queries_c, queries_i : Tensor  (Q,)
        Clique and song IDs for the queries.
    candidates_c, candidates_i : Tensor  (D,)
        Clique and song IDs for the candidates.
    recall_ks : tuple of int
        Which K values to evaluate Recall@K for.
    verbose : bool
        Print progress.

    Returns
    -------
    Same dict as :func:`evaluate`.
    """
    Q = len(queries_i)
    ap_list  = []
    r1_list  = []
    nar_list = []
    rk_lists = {k: [] for k in recall_ks}

    for n in range(Q):
        if verbose and n % 100 == 0:
            print(f"  query {n}/{Q} …", flush=True)

        dist     = dist_matrix[n]
        is_match = candidates_c == queries_c[n]

        # exclude self
        is_self  = candidates_i == queries_i[n]
        dist     = torch.where(is_self, torch.full_like(dist, torch.inf), dist)
        is_match = torch.where(is_self, torch.zeros_like(is_match), is_match)

        ap_list.append(_average_precision(dist, is_match))
        r1_list.append(_rank_of_first_correct(dist, is_match))
        nar_list.append(_normalised_average_rank(dist, is_match))
        for k in recall_ks:
            rk_lists[k].append(_recall_at_k(dist, is_match, k))

    ap  = torch.stack(ap_list)
    r1  = torch.stack(r1_list)
    nar = torch.stack(nar_list)

    results: dict = {
        "mAP": ap.mean().item(),
        "MR1": r1.mean().item(),
        "NAR": nar.mean().item(),
        "ap":  ap,
        "r1":  r1,
        "nar": nar,
    }
    for k in recall_ks:
        rk = torch.stack(rk_lists[k])
        results[f"R@{k}"] = rk.mean().item()
        results[f"r@{k}"] = rk

    return results


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------


def print_results(results: dict, recall_ks: tuple[int, ...] = (1, 10, 100)) -> None:
    """
    Print a tidy summary table of the evaluation results.

    Parameters
    ----------
    results : dict
        Output of :func:`evaluate` or :func:`evaluate_from_distances`.
    recall_ks : tuple of int
        The K values that were evaluated (must match those used when computing
        results).
    """
    print("=" * 42)
    print(f"  {'Metric':<20} {'Value':>10}")
    print("-" * 42)
    print(f"  {'mAP':<20} {results['mAP']:>10.4f}")
    print(f"  {'MR1':<20} {results['MR1']:>10.2f}")
    print(f"  {'NAR (%)':<20} {results['NAR']:>10.2f}")
    for k in recall_ks:
        key = f"R@{k}"
        if key in results:
            print(f"  {key:<20} {results[key]:>10.4f}")
    print("=" * 42)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _compute_distances(
    model,
    query_idx: int,
    queries_z: Tensor,
    candidates_z: Tensor,
    queries_m,
    candidates_m,
    redux_strategy,
    batch_size_candidates: int | None,
) -> Tensor:
    """Compute distances from query *query_idx* to all candidates."""
    q_z = queries_z[query_idx : query_idx + 1].float()
    q_m = queries_m[query_idx : query_idx + 1] if queries_m is not None else None

    if batch_size_candidates is None or batch_size_candidates >= len(candidates_z):
        dist = model.distances(
            q_z, candidates_z.float(),
            qmask=q_m, cmask=candidates_m,
            redux_strategy=redux_strategy,
        ).squeeze(0)
    else:
        chunks = []
        D = len(candidates_z)
        for start in range(0, D, batch_size_candidates):
            end = min(start + batch_size_candidates, D)
            c_m = candidates_m[start:end] if candidates_m is not None else None
            chunk = model.distances(
                q_z, candidates_z[start:end].float(),
                qmask=q_m, cmask=c_m,
                redux_strategy=redux_strategy,
            ).squeeze(0)
            chunks.append(chunk)
        dist = torch.cat(chunks, dim=-1)

    return dist


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal self-contained demo.  Run with:
        python cover_eval.py
    """
    import random
    torch.manual_seed(0)

    # --- toy model stub ---
    class ToyModel:
        """A dummy model that computes cosine distance."""
        def distances(self, q, c, qmask=None, cmask=None, redux_strategy=None):
            # q: (1, S, C), c: (D, S, C)  → (1, D)
            q_mean = q.mean(dim=1)                     # (1, C)
            c_mean = c.mean(dim=1)                     # (D, C)
            q_norm = torch.nn.functional.normalize(q_mean, dim=-1)
            c_norm = torch.nn.functional.normalize(c_mean, dim=-1)
            cosine_sim = (q_norm @ c_norm.T)           # (1, D)
            return 1.0 - cosine_sim                    # cosine distance

        def eval(self):
            pass

    # --- fake data: 20 songs, 4 cliques of 5 covers each ---
    Q, D, S, C = 20, 20, 4, 64
    clique_ids  = torch.arange(Q) // 5                 # [0,0,0,0,0, 1,1,1,1,1, ...]
    song_ids    = torch.arange(Q)
    embeddings  = torch.randn(Q, S, C)
    # Add clique signal: songs in the same clique share a base vector
    for i in range(Q):
        embeddings[i] += 5.0 * torch.randn(1, 1, C).expand(1, S, C)[0] * (clique_ids[i].float() / 4)

    model = ToyModel()
    print("Running evaluate() …")
    results = evaluate(
        model,
        queries_z=embeddings,   candidates_z=embeddings,
        queries_c=clique_ids,   candidates_c=clique_ids,
        queries_i=song_ids,     candidates_i=song_ids,
        recall_ks=(1, 5, 10),
        verbose=True,
    )
    print_results(results, recall_ks=(1, 5, 10))

    print("\nRunning evaluate_from_distances() with a random distance matrix …")
    dist_mat = torch.rand(Q, D)
    results2 = evaluate_from_distances(
        dist_mat,
        queries_c=clique_ids, queries_i=song_ids,
        candidates_c=clique_ids, candidates_i=song_ids,
        recall_ks=(1, 10),
    )
    print_results(results2, recall_ks=(1, 10))
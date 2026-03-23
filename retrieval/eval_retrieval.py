"""eval_retrieval.py
==================
Full retrieval + cover-song evaluation pipeline.

Loads embeddings from *first_embeddings.txt* (database) and
*second_embeddings.txt* (queries), computes a full pairwise L2 distance
matrix, and reports mAP / MR1 / NAR / R@K using the metrics in
``eval/eval.py``.

Usage
-----
    python eval_retrieval.py \\
        --first-list  extractor/first_embeddings.txt \\
        --second-list extractor/second_embeddings.txt \\
        --metadata-json /data/discogs_test_subset.json \\
        [--dim 1024] [--k 1 10 100] [--verbose]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from sibling packages
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent

# retrieval/ package
sys.path.insert(0, str(_ROOT))
from retrieval import FaissRetrievalIndex  # noqa: E402

# eval/eval.py — loaded via importlib to avoid shadowing the builtin `eval`
_eval_spec = importlib.util.spec_from_file_location(
    "cover_eval", _ROOT / "eval" / "eval.py"
)
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)
evaluate_from_distances = _eval_mod.evaluate_from_distances
print_results = _eval_mod.print_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_path_list(list_path: Path) -> list[Path]:
    lines = list_path.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def load_metadata(
    metadata_path: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """Parse metadata JSON into per-track clique and song integer IDs.

    Expected JSON shape::

        {"<version_id>": [{"youtube_id": "..."}, ...], ...}

    Returns
    -------
    clique_map : dict[str, int]
        youtube_id  →  integer clique/version-group ID
    song_map : dict[str, int]
        youtube_id  →  integer song ID (unique per track)
    """
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clique_map: dict[str, int] = {}
    song_map: dict[str, int] = {}
    version_to_int: dict[str, int] = {}
    clique_counter = 0
    song_counter = 0

    for version_id, versions in data.items():
        if not isinstance(versions, list):
            continue
        if version_id not in version_to_int:
            version_to_int[version_id] = clique_counter
            clique_counter += 1
        clique_int = version_to_int[version_id]

        for item in versions:
            if not isinstance(item, dict):
                continue
            youtube_id = item.get("youtube_id")
            if not youtube_id:
                continue
            yt = str(youtube_id)
            clique_map[yt] = clique_int
            if yt not in song_map:
                song_map[yt] = song_counter
                song_counter += 1

    return clique_map, song_map


def _strip_row_suffix(embedding_id: str) -> str:
    """Remove '#row' suffix produced for multi-row embedding files."""
    return embedding_id.split("#", 1)[0]


def load_embeddings(
    files: list[Path], expected_dim: int
) -> tuple[np.ndarray, list[str]]:
    """Load all embedding files into a single (N, D) array with matching IDs."""
    arrays: list[np.ndarray] = []
    ids: list[str] = []

    for path in files:
        arr = np.asarray(
            FaissRetrievalIndex._load_embedding_file(path), dtype=np.float32
        )
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"Unsupported shape {arr.shape} in {path}")
        if arr.shape[1] != expected_dim:
            raise ValueError(
                f"Dimension mismatch in {path}: got {arr.shape[1]}, expected {expected_dim}"
            )

        base_id = path.stem
        if arr.shape[0] == 1:
            ids.append(base_id)
        else:
            ids.extend(f"{base_id}#{i}" for i in range(arr.shape[0]))
        arrays.append(arr)

    return np.concatenate(arrays, axis=0), ids


def build_id_tensors(
    emb_ids: list[str],
    clique_map: dict[str, int],
    song_map: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Map embedding IDs to integer tensors, skipping IDs absent in metadata.

    Returns
    -------
    clique_ids : LongTensor (M,)
    song_ids   : LongTensor (M,)
    valid_rows : list[int]  — indices into the original embedding matrix
    """
    clique_ids: list[int] = []
    song_ids: list[int] = []
    valid_rows: list[int] = []

    for idx, eid in enumerate(emb_ids):
        base = _strip_row_suffix(eid)
        if base in clique_map:
            clique_ids.append(clique_map[base])
            song_ids.append(song_map[base])
            valid_rows.append(idx)

    return (
        torch.tensor(clique_ids, dtype=torch.long),
        torch.tensor(song_ids, dtype=torch.long),
        valid_rows,
    )


def filter_queries_with_positives(
    qry_c: torch.Tensor,
    qry_i: torch.Tensor,
    cand_c: torch.Tensor,
    cand_i: torch.Tensor,
) -> list[int]:
    """Return indices of queries that have ≥1 positive candidate (excluding self)."""
    valid: list[int] = []
    for q in range(len(qry_i)):
        same_clique = cand_c == qry_c[q]
        not_self = cand_i != qry_i[q]
        if (same_clique & not_self).any():
            valid.append(q)
    return valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval + cover-song evaluation (mAP, MR1, NAR, R@K)"
    )
    parser.add_argument(
        "--first-list",
        type=Path,
        default=Path("extractor/first_embeddings.txt"),
        help="Path to list of database embedding files.",
    )
    parser.add_argument(
        "--second-list",
        type=Path,
        default=Path("extractor/second_embeddings.txt"),
        help="Path to list of query embedding files.",
    )
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimension.")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("/data/discogs_test_subset.json"),
        help="Metadata JSON mapping version_id → [{youtube_id, ...}].",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 10, 100],
        metavar="K",
        help="R@K values to compute (default: 1 10 100).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-query progress."
    )
    args = parser.parse_args()

    # ---- load file lists -------------------------------------------------------
    first_files = read_path_list(args.first_list)
    second_files = read_path_list(args.second_list)

    if not first_files:
        raise ValueError(f"No embeddings found in {args.first_list}")
    if not second_files:
        raise ValueError(f"No embeddings found in {args.second_list}")

    print(f"Database files : {len(first_files)}")
    print(f"Query files    : {len(second_files)}")

    # ---- metadata --------------------------------------------------------------
    print(f"Loading metadata from {args.metadata_json} …")
    clique_map, song_map = load_metadata(args.metadata_json)
    n_cliques = len(set(clique_map.values()))
    print(f"  {len(clique_map)} tracks  |  {n_cliques} cliques")

    # ---- load embeddings -------------------------------------------------------
    print("Loading database embeddings …")
    db_vecs, db_ids = load_embeddings(first_files, args.dim)
    print(f"  {db_vecs.shape[0]} vectors  shape={db_vecs.shape}")

    print("Loading query embeddings …")
    q_vecs, q_ids = load_embeddings(second_files, args.dim)
    print(f"  {q_vecs.shape[0]} vectors  shape={q_vecs.shape}")

    # ---- map IDs to tensors, filter to metadata-known tracks ------------------
    cand_c, cand_i, cand_valid = build_id_tensors(db_ids, clique_map, song_map)
    qry_c, qry_i, qry_valid = build_id_tensors(q_ids, clique_map, song_map)

    n_skip_db = db_vecs.shape[0] - len(cand_valid)
    n_skip_q = q_vecs.shape[0] - len(qry_valid)
    if n_skip_db:
        print(f"  Warning: {n_skip_db} database vector(s) skipped (not in metadata)")
    if n_skip_q:
        print(f"  Warning: {n_skip_q} query vector(s) skipped (not in metadata)")

    db_vecs = db_vecs[cand_valid]
    q_vecs = q_vecs[qry_valid]

    # ---- drop queries that have no positive candidate (after self-exclusion) --
    usable_q = filter_queries_with_positives(qry_c, qry_i, cand_c, cand_i)
    n_no_pos = len(qry_valid) - len(usable_q)
    if n_no_pos:
        print(
            f"  Warning: {n_no_pos} query/queries dropped (no positive candidate in DB)"
        )

    qry_c = qry_c[usable_q]
    qry_i = qry_i[usable_q]
    q_vecs = q_vecs[usable_q]

    if len(usable_q) == 0:
        raise ValueError("No usable queries — nothing to evaluate.")
    if len(cand_valid) == 0:
        raise ValueError("Database is empty — nothing to search.")

    print(f"Queries  : {len(usable_q)}")
    print(f"Database : {len(cand_valid)}")

    # ---- full pairwise L2 distance matrix (Q × D) -----------------------------
    print("Computing pairwise L2 distance matrix …")
    q_t = torch.from_numpy(q_vecs)    # (Q, D)
    db_t = torch.from_numpy(db_vecs)  # (D, D)
    dist_matrix = torch.cdist(q_t, db_t, p=2)  # (Q, D)
    print(f"  Distance matrix shape: {list(dist_matrix.shape)}")

    # ---- evaluate --------------------------------------------------------------
    recall_ks = tuple(args.k)
    print("Running evaluation …")
    results = evaluate_from_distances(
        dist_matrix=dist_matrix,
        queries_c=qry_c,
        queries_i=qry_i,
        candidates_c=cand_c,
        candidates_i=cand_i,
        recall_ks=recall_ks,
        verbose=args.verbose,
    )

    print_results(results, recall_ks=recall_ks)


if __name__ == "__main__":
    main()

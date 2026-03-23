"""Retrieval evaluation entrypoint.

Supports two label sources:
- --metadata-json: version_id -> [{youtube_id, ...}]
- --labels-json: embedding_stem -> {work_id, song_id}

Loads embeddings from first/second list files, computes full pairwise L2 distances,
and reports mAP / MR1 / NAR / R@K.
"""

from __future__ import annotations

import argparse
import importlib.util
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
from retrieval import FaissRetrievalIndex  # noqa: E402

_eval_spec = importlib.util.spec_from_file_location("cover_eval", _ROOT / "eval" / "eval.py")
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)
evaluate_from_distances = _eval_mod.evaluate_from_distances
print_results = _eval_mod.print_results


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise ImportError(
            "PyTorch is required for retrieval/eval_retrieval.py. "
            "Install dependencies in the active environment or run via docker compose retrieval service."
        ) from exc


def read_path_list(list_path: Path) -> list[Path]:
    lines = list_path.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def load_metadata(metadata_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clique_map: dict[str, int] = {}
    song_map: dict[str, int] = {}
    version_to_int: dict[str, int] = {}

    for version_id, versions in data.items():
        if not isinstance(versions, list):
            continue
        if version_id not in version_to_int:
            version_to_int[version_id] = len(version_to_int)
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
                song_map[yt] = len(song_map)

    return clique_map, song_map


def load_labels_json(labels_json: Path) -> tuple[dict[str, int], dict[str, int]]:
    with labels_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("labels-json must be a JSON object")

    clique_map: dict[str, int] = {}
    song_map: dict[str, int] = {}
    clique_to_int: dict[str, int] = {}
    song_to_int: dict[str, int] = {}

    for emb_id, payload in data.items():
        if not isinstance(payload, dict):
            continue
        work_id = payload.get("work_id")
        song_id = payload.get("song_id")
        if not work_id or not song_id:
            continue

        work_id = str(work_id)
        song_id = str(song_id)

        if work_id not in clique_to_int:
            clique_to_int[work_id] = len(clique_to_int)
        if song_id not in song_to_int:
            song_to_int[song_id] = len(song_to_int)

        clique_map[str(emb_id)] = clique_to_int[work_id]
        song_map[str(emb_id)] = song_to_int[song_id]

    return clique_map, song_map


def _strip_row_suffix(embedding_id: str) -> str:
    return embedding_id.split("#", 1)[0]


def load_embeddings(files: list[Path], expected_dim: int) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    ids: list[str] = []

    for path in files:
        arr = np.asarray(FaissRetrievalIndex._load_embedding_file(path), dtype=np.float32)
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

    if not arrays:
        raise ValueError("No embedding arrays loaded")

    return np.concatenate(arrays, axis=0), ids


def build_id_tensors(
    emb_ids: list[str],
    clique_map: dict[str, int],
    song_map: dict[str, int],
) -> tuple[Any, Any, list[int]]:
    torch = _require_torch()
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
    qry_c: Any,
    qry_i: Any,
    cand_c: Any,
    cand_i: Any,
) -> list[int]:
    valid: list[int] = []
    for q in range(len(qry_i)):
        same_clique = cand_c == qry_c[q]
        not_self = cand_i != qry_i[q]
        if (same_clique & not_self).any():
            valid.append(q)
    return valid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval + cover-song evaluation")
    parser.add_argument("--first-list", type=Path, required=True)
    parser.add_argument("--second-list", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--labels-json", type=Path, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def evaluate_from_args(args: argparse.Namespace) -> dict[str, object]:
    torch = _require_torch()
    if args.metadata_json is None and args.labels_json is None:
        raise ValueError("Provide either --metadata-json or --labels-json")
    if args.metadata_json is not None and args.labels_json is not None:
        raise ValueError("Use only one label source: --metadata-json or --labels-json")

    first_files = read_path_list(args.first_list)
    second_files = read_path_list(args.second_list)
    if not first_files:
        raise ValueError(f"No embeddings found in {args.first_list}")
    if not second_files:
        raise ValueError(f"No embeddings found in {args.second_list}")

    if args.labels_json is not None:
        clique_map, song_map = load_labels_json(args.labels_json)
        label_source = "labels-json"
    else:
        clique_map, song_map = load_metadata(args.metadata_json)
        label_source = "metadata-json"

    db_vecs, db_ids = load_embeddings(first_files, args.dim)
    q_vecs, q_ids = load_embeddings(second_files, args.dim)

    cand_c, cand_i, cand_valid = build_id_tensors(db_ids, clique_map, song_map)
    qry_c, qry_i, qry_valid = build_id_tensors(q_ids, clique_map, song_map)

    n_skip_db = db_vecs.shape[0] - len(cand_valid)
    n_skip_q = q_vecs.shape[0] - len(qry_valid)

    db_vecs = db_vecs[cand_valid]
    q_vecs = q_vecs[qry_valid]

    usable_q = filter_queries_with_positives(qry_c, qry_i, cand_c, cand_i)
    n_no_pos = len(qry_valid) - len(usable_q)

    qry_c = qry_c[usable_q]
    qry_i = qry_i[usable_q]
    q_vecs = q_vecs[usable_q]

    if len(usable_q) == 0:
        raise ValueError("No usable queries -- nothing to evaluate")
    if len(cand_valid) == 0:
        raise ValueError("Database is empty -- nothing to evaluate")

    dist_matrix = torch.cdist(torch.from_numpy(q_vecs), torch.from_numpy(db_vecs), p=2)
    recall_ks = tuple(args.k)
    results = evaluate_from_distances(
        dist_matrix=dist_matrix,
        queries_c=qry_c,
        queries_i=qry_i,
        candidates_c=cand_c,
        candidates_i=cand_i,
        recall_ks=recall_ks,
        verbose=args.verbose,
    )

    return {
        "metrics": {
            "mAP": float(results["mAP"]),
            "MR1": float(results["MR1"]),
            "NAR": float(results["NAR"]),
            **{f"R@{k}": float(results[f"R@{k}"]) for k in recall_ks},
        },
        "details": {
            "label_source": label_source,
            "num_database_files": len(first_files),
            "num_query_files": len(second_files),
            "num_database_vectors": int(db_vecs.shape[0]),
            "num_query_vectors": int(q_vecs.shape[0]),
            "num_cliques": int(len(set(clique_map.values()))),
            "num_skipped_database_vectors": int(n_skip_db),
            "num_skipped_query_vectors": int(n_skip_q),
            "num_dropped_queries_no_positive": int(n_no_pos),
            "recall_ks": list(recall_ks),
        },
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output = evaluate_from_args(args)
    print_results(output["metrics"], recall_ks=tuple(output["details"]["recall_ks"]))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=True)
        print(f"Saved evaluation JSON to: {args.output_json}")


if __name__ == "__main__":
    main()

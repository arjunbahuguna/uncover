"""
End-to-end cover retrieval pipeline orchestrator.
Supports Windows GPU environments with intelligent Docker environment checks.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PathLike = str | Path
REPO_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = "docker-compose.windows-gpu.yml"


@dataclass
class SelectedWorkPair:
    """Represents a pair of audio files (index and query) for a specific work."""
    work_id: str
    index_audio: Path
    query_audio: Path


@dataclass
class QueryItem:
    """Represents a single audio item used in the retrieval process."""
    audio_path: Path
    work_id: str
    song_id: str
    source: str


def _extract_path_from_item(item: Any) -> Path | None:
    """
    Extracts a valid file path from various input formats (string or dictionary).

    Args:
        item: Can be a string path or a dictionary containing path keys.

    Returns:
        A resolved Path object if found, otherwise None.
    """
    raw_val = None

    if isinstance(item, str):
        raw_val = item.strip()
    elif isinstance(item, dict):
        # Check common keys for file paths
        for key in ("path", "audio_path", "recording_path", "file_path", "filepath"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                raw_val = val.strip()
                break

    if not raw_val:
        return None

    # Handle absolute vs relative paths
    if raw_val.startswith("/"):
        return Path(raw_val)
    return (REPO_ROOT / raw_val).resolve()


def load_work_to_paths(json_path: Path) -> dict[str, list[Path]]:
    """
    Loads a JSON file mapping work IDs to lists of audio file paths.

    Requirements:
    - Input must be a JSON object.
    - Only works with 2 or more valid audio paths are included.
    - Duplicate paths within a work are removed.
    """
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object: {work_id: [recordings...]}")

    work_to_paths: dict[str, list[Path]] = {}

    for raw_work_id, items in data.items():
        work_id = str(raw_work_id)
        if not isinstance(items, list):
            continue

        seen: set[str] = set()
        paths: list[Path] = []

        for item in items:
            candidate = _extract_path_from_item(item)
            if candidate is None:
                continue

            cand_str = candidate.as_posix()
            if cand_str not in seen:
                seen.add(cand_str)
                paths.append(candidate)

        # Only include works that have at least 2 distinct audio files
        if len(paths) >= 2:
            work_to_paths[work_id] = paths

    return work_to_paths


def select_pairs(work_to_paths: dict[str, list[Path]], seed: int) -> list[SelectedWorkPair]:
    """
    Selects one index audio and one query audio for each work ID.

    Args:
        work_to_paths: Dictionary mapping work IDs to lists of paths.
        seed: Random seed for reproducibility.

    Returns:
        List of SelectedWorkPair objects.
    """
    rng = random.Random(seed)
    selected: list[SelectedWorkPair] = []

    # Sort keys to ensure deterministic order before sampling
    for work_id in sorted(work_to_paths.keys()):
        candidates = work_to_paths[work_id]
        # Randomly sample two distinct paths
        idx_audio, qry_audio = rng.sample(candidates, 2)
        selected.append(
            SelectedWorkPair(
                work_id=work_id,
                index_audio=idx_audio,
                query_audio=qry_audio,
            )
        )
    return selected


def ensure_unique_stems(paths: list[Path]) -> None:
    """
    Ensures all file stems (filenames without extension) are unique.

    This is critical because embedding IDs are often derived from file stems.
    Raises ValueError if duplicates are found.
    """
    stems: dict[str, Path] = {}
    duplicates: list[tuple[str, Path, Path]] = []

    for path in paths:
        stem = path.stem
        if stem in stems and stems[stem] != path:
            duplicates.append((stem, stems[stem], path))
        else:
            stems[stem] = path

    if duplicates:
        lines = ["Found duplicate file stems. Embedding IDs are stem-based and would collide:"]
        for stem, p1, p2 in duplicates[:10]:
            lines.append(f"  stem='{stem}' -> '{p1}' and '{p2}'")
        raise ValueError("\n".join(lines))


def write_path_list(paths: list[Path], output_txt: Path) -> None:
    """Writes a list of paths to a text file, one per line."""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path.as_posix()}\n")


def host_path_to_container(path: Path) -> Path:
    """
    Converts a host machine path to a Docker container path.

    Assumes the repository root is mounted at '/app' inside the container.
    If the path is outside the repo, returns the original path string.
    """
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
        return Path("/app") / rel.as_posix()
    except ValueError:
        return Path(path.as_posix())


def ensure_docker_env(service: str) -> None:
    """
    Intelligently checks if the required Docker image exists.
    Builds the image only if it is missing to save time.

    Args:
        service: The service name defined in docker-compose.yml.
    """
    print(f"\n>>> [System Check] Verifying environment for '{service}'...")
    try:
        # 1. Get the final image name from the compose configuration
        cfg_cmd = ["docker", "compose", "-f", COMPOSE_FILE, "config", "--format", "json"]
        cfg_res = subprocess.run(cfg_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        config_data = json.loads(cfg_res.stdout)

        image_name = config_data.get("services", {}).get(service, {}).get("image")
        if not image_name:
            # Fallback guess if image name isn't explicitly defined
            image_name = f"{REPO_ROOT.name.lower()}-{service}"

        # 2. Check if the image exists locally
        img_cmd = ["docker", "images", "-q", image_name]
        img_res = subprocess.run(img_cmd, capture_output=True, text=True)

        if img_res.stdout.strip():
            print(f">>> [Ready] Image '{image_name}' found. Skipping build.")
        else:
            print(
                f">>> [Missing] Image '{image_name}' not found. Triggering automatic build (this may take a few minutes)...")
            build_cmd = ["docker", "compose", "-f", COMPOSE_FILE, "build", service]
            subprocess.run(build_cmd, cwd=REPO_ROOT, check=True)
            print(f">>> [Complete] Environment for '{service}' is ready.")

    except Exception as e:
        print(f">>> [Warning] Could not parse image status. Delegating to Docker default behavior. ({str(e)})")


def run_embedding_extractor_docker(input_list: Path, model: str, output_dir: Path) -> None:
    """
    Runs the embedding extraction process inside a Docker container.

    Args:
        input_list: Path to a text file listing input audio files.
        model: The model to use ('clews' or 'discogs-vinet').
        output_dir: Directory to save extracted embeddings.
    """
    service_name = "clews" if model == "clews" else "discogs-vinet"

    # Ensure the Docker environment is ready
    ensure_docker_env(service_name)

    # Convert paths to be relative to repo root for Docker mounting
    input_rel = input_list.resolve().relative_to(REPO_ROOT).as_posix()
    output_rel = output_dir.resolve().relative_to(REPO_ROOT).as_posix()

    cmd = ["docker", "compose", "-f", COMPOSE_FILE, "run", "--rm"]

    if model == "clews":
        cmd.extend([
            "clews", "python", "extractor/extractor_gpu_clew.py",
            "--input", input_rel,
            "--checkpoint", "models/clews/checkpoints/clews/dvi-clews/checkpoint_best.ckpt",
            "--output-path", output_rel
        ])
    elif model == "discogs-vinet":
        cmd.extend([
            "discogs-vinet", "python", "extractor/extractor_vinet_gpu.py",
            "--input", input_rel,
            "--output-path", output_rel
        ])
    else:
        raise ValueError(f"Unsupported model: {model}")

    print("\n>>> Starting GPU-accelerated extraction container...")
    process = subprocess.run(cmd, cwd=REPO_ROOT, check=False)

    if process.returncode != 0:
        raise RuntimeError("Docker extraction failed. Please check the logs above.")


def run_retrieval_evaluation_docker(
        first_list: Path, second_list: Path, labels_json: Path,
        dim: int, recall_ks: tuple[int, ...], output_json: Path
) -> None:
    """
    Runs the retrieval evaluation script inside a Docker container.

    Args:
        first_list: Path to index embeddings list.
        second_list: Path to query embeddings list.
        labels_json: Path to ground truth labels.
        dim: Embedding dimension.
        recall_ks: Tuple of K values for Recall@K calculation.
        output_json: Path to save evaluation results.
    """
    # Ensure the Docker environment is ready
    ensure_docker_env("retrieval")

    cmd = [
        "docker", "compose", "-f", COMPOSE_FILE, "run", "--rm", "retrieval",
        "python", "retrieval/eval_retrieval.py",
        "--first-list", first_list.as_posix(),
        "--second-list", second_list.as_posix(),
        "--labels-json", labels_json.as_posix(),
        "--dim", str(dim),
        "--k", *[str(k) for k in recall_ks],
        "--output-json", output_json.as_posix(),
    ]

    print("\n>>> Starting retrieval evaluation container...")
    process = subprocess.run(cmd, cwd=REPO_ROOT, check=False)

    if process.returncode != 0:
        raise RuntimeError("Docker retrieval evaluation failed.")


def embedding_path_for_audio(audio_path: Path, embeddings_dir: Path, model: str) -> Path:
    """Generates the expected output path for an embedding based on the model type."""
    suffix = ".pt" if model == "clews" else ".npy"
    return embeddings_dir / f"{audio_path.stem}{suffix}"


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """
    Orchestrates the full pipeline: loading data, extracting embeddings, and evaluating retrieval.

    Returns:
        A dictionary containing metrics and details of the run.
    """
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Runtime directory for Docker volume mounting
    docker_runtime_root = REPO_ROOT / "extractor" / "pipeline_runtime" / output_dir.name
    docker_runtime_root.mkdir(parents=True, exist_ok=True)

    # Load and validate data
    work_to_paths = load_work_to_paths(args.input_json)
    if not work_to_paths:
        raise ValueError("No valid data found in JSON! Please check the format.")

    # Select pairs for testing
    selected_pairs = select_pairs(work_to_paths=work_to_paths, seed=args.seed)

    index_items, query_items = [], []
    for pair in selected_pairs:
        index_items.append(QueryItem(pair.index_audio, pair.work_id, f"{pair.work_id}::index", "index"))
        query_items.append(QueryItem(pair.query_audio, pair.work_id, f"{pair.work_id}::query", "query"))

    all_audio_paths = [item.audio_path for item in index_items + query_items]
    ensure_unique_stems(all_audio_paths)

    audio_list_path = docker_runtime_root / "all_audio_paths.txt"

    # Determine which embeddings need to be generated
    expected_embeddings = [embedding_path_for_audio(p, embeddings_dir, args.embedding_model) for p in all_audio_paths]
    missing_audio_paths = [p for p, emb in zip(all_audio_paths, expected_embeddings) if not emb.is_file()]

    if args.skip_embedding_extraction:
        if missing_audio_paths:
            raise FileNotFoundError(
                f"--skip-embedding-extraction was set, but {len(missing_audio_paths)} embedding files are missing in {embeddings_dir}.")
        print("\n>>> Skipping extraction as requested (--skip-embedding-extraction).")
    else:
        if missing_audio_paths:
            write_path_list(missing_audio_paths, audio_list_path)
            print(f"\n>>> Found {len(missing_audio_paths)} audio files requiring feature extraction.")
            run_embedding_extractor_docker(audio_list_path, args.embedding_model, embeddings_dir)
        else:
            print("\n>>> All embedding files exist. Proceeding directly to evaluation.")

    # Prepare file lists for the evaluation script
    first_list = output_dir / "first_embeddings.txt"
    second_list = output_dir / "second_embeddings.txt"

    write_path_list(
        [embedding_path_for_audio(item.audio_path, embeddings_dir, args.embedding_model) for item in index_items],
        first_list)
    write_path_list(
        [embedding_path_for_audio(item.audio_path, embeddings_dir, args.embedding_model) for item in query_items],
        second_list)

    # Prepare Docker-specific paths (mounted volumes)
    first_list_docker = docker_runtime_root / "first_embeddings_docker.txt"
    second_list_docker = docker_runtime_root / "second_embeddings_docker.txt"

    write_path_list(
        [host_path_to_container(p) for p in
         [embedding_path_for_audio(item.audio_path, embeddings_dir, args.embedding_model) for item in index_items]],
        first_list_docker)

    write_path_list(
        [host_path_to_container(p) for p in
         [embedding_path_for_audio(item.audio_path, embeddings_dir, args.embedding_model) for item in query_items]],
        second_list_docker)

    # Generate labels JSON
    labels_json = output_dir / "embedding_labels.json"
    with labels_json.open("w", encoding="utf-8") as f:
        json.dump(
            {item.audio_path.stem: {"work_id": item.work_id, "song_id": item.song_id} for item in
             index_items + query_items},
            f, indent=2
        )

    # Run Evaluation
    eval_results_json = output_dir / "eval_results.json"
    run_retrieval_evaluation_docker(
        host_path_to_container(first_list_docker),
        host_path_to_container(second_list_docker),
        host_path_to_container(labels_json),
        args.embedding_dim,
        tuple(int(x) for x in args.recall_ks.split(",")),
        host_path_to_container(eval_results_json)
    )

    # Load and return results
    with eval_results_json.open("r", encoding="utf-8") as f:
        eval_output = json.load(f)

    return {
        "works_used": len(selected_pairs),
        "embedding_model": args.embedding_model,
        "metrics": eval_output["metrics"],
        "details": eval_output["details"],
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end cover retrieval pipeline.")
    parser.add_argument("--input-json", type=Path, required=True, help="Path to input JSON file.")
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline_runs/latest"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--skip-embedding-extraction", action="store_true",
                        help="Skip embedding generation if files exist.")
    parser.add_argument("--embedding-model", choices=["clews", "discogs-vinet"], required=True, help="Model to use.")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Dimension of the embeddings.")
    parser.add_argument("--recall-ks", type=str, default="1,10,100", help="Comma-separated K values for Recall@K.")

    args = parser.parse_args()

    report = run_pipeline(args)

    print("\n============== Retrieval Evaluation Results ==============")
    print(f"Works used: {report['works_used']}")
    print(f"mAP (Mean Average Precision): {report['metrics']['mAP']:.6f}")
    print(f"MR1 (Median Rank 1 / Top-1 Recall): {report['metrics']['MR1']:.6f}")

    for key in sorted([k for k in report["metrics"].keys() if k.startswith("R@")]):
        print(f"{key}: {report['metrics'][key]:.6f}")


if __name__ == "__main__":
    main()
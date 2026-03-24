"""
End-to-end Automated Cover Retrieval Pipeline Orchestrator.
Native Python Audio Pre-processing + Dockerized GPU Extraction & Evaluation.
"""

from __future__ import annotations
import concurrent.futures
import json
import random
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import concurrent.futures
# Import audio processing libraries
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Configuration: Host root directory for the dataset (Windows path example)
DATASET_HOST_ROOT = Path(r"E:\Master\Master_Util\datasets_cover\discogs")
# Configuration: Container root directory for the dataset (Linux path inside Docker)
DATASET_CONTAINER_ROOT = Path("/data/discogs")

PathLike = str | Path
REPO_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = "docker-compose.windows-gpu.yml"
CONFIG_FILE = "config_test.json"


@dataclass
class SelectedWorkPair:
    """Represents a pair of audio files (index and query) belonging to the same musical work."""
    work_id: str
    index_audio: Path
    query_audio: Path


@dataclass
class QueryItem:
    """Represents a single audio item used in the retrieval pipeline."""
    audio_path: Path
    work_id: str
    song_id: str
    source: str


def load_work_to_paths(json_path: Path) -> dict[str, list[Path]]:
    """
    Loads a JSON mapping of work IDs to lists of audio file paths.
    Filters out works with fewer than 2 audio files and removes duplicate paths.
    Handles path conversion from container paths to host paths if necessary.
    """
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

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

        # Only keep works that have at least two distinct audio files
        if len(paths) >= 2:
            work_to_paths[work_id] = paths

    return work_to_paths


def _extract_path_from_item(item: Any) -> Path | None:
    """
    Extracts a file path from various input formats (string or dictionary).
    Performs critical path redirection: converts container paths (/data/discogs/...)
    to actual host machine paths.
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

    # --- Critical Fix: Path Redirection ---
    # If the JSON contains a container path (e.g., /data/discogs/...),
    # convert it to the host machine's real path.
    if raw_val.startswith("/data/discogs"):
        rel_path = raw_val.replace("/data/discogs/", "").replace("/data/discogs", "")
        rel_path = rel_path.lstrip("/")
        # Construct host path: E:\Master\...\rel_path
        return (DATASET_HOST_ROOT / rel_path).resolve()

    # Handle normal absolute paths
    if raw_val.startswith("/"):
        return Path(raw_val)

    # Handle relative paths (relative to the repository root)
    return (REPO_ROOT / raw_val).resolve()


def host_path_to_container(path: Path) -> Path:
    """
    Converts a host machine path to the corresponding path inside the Docker container.
    - Maps repository code to /app
    - Maps dataset directory to /data/discogs
    - Falls back to converting backslashes to forward slashes for other paths.
    """
    resolved_path = path.resolve()

    # Case A: Path is within the project code directory
    try:
        rel = resolved_path.relative_to(REPO_ROOT.resolve())
        return Path("/app") / rel.as_posix()
    except ValueError:
        pass

    # Case B: Path is within the dataset directory
    try:
        rel = resolved_path.relative_to(DATASET_HOST_ROOT.resolve())
        return Path("/data/discogs") / rel.as_posix()
    except ValueError:
        pass

    # Case C: Other paths (convert Windows separators to Unix style)
    return Path(str(resolved_path).replace('\\', '/'))


def select_pairs(work_to_paths: dict[str, list[Path]], seed: int) -> list[SelectedWorkPair]:
    """
    Selects one index audio and one query audio randomly for each work ID.
    Uses a fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    selected: list[SelectedWorkPair] = []

    # Sort keys to ensure deterministic order before sampling
    for work_id in sorted(work_to_paths.keys()):
        candidates = work_to_paths[work_id]
        # Randomly sample 2 distinct files
        idx_audio, qry_audio = rng.sample(candidates, 2)
        selected.append(SelectedWorkPair(work_id, idx_audio, qry_audio))

    return selected


def ensure_unique_stems(paths: list[Path]) -> None:
    """
    Validates that all file stems (filenames without extension) are unique.
    Raises an error if duplicates are found to prevent embedding ID collisions.
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
        raise ValueError("Found duplicate file stems! Embedding IDs would collide.")


def write_path_list(paths: list[Path], output_txt: Path) -> None:
    """Writes a list of file paths to a text file, one per line."""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path.as_posix()}\n")


def ensure_docker_env(service: str) -> None:
    """
    Checks if the required Docker image exists. If not, automatically builds it.
    """
    try:
        # Get docker compose configuration
        cfg_cmd = ["docker", "compose", "-f", COMPOSE_FILE, "config", "--format", "json"]
        cfg_res = subprocess.run(cfg_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        config_data = json.loads(cfg_res.stdout)

        image_name = config_data.get("services", {}).get(service, {}).get("image")
        if not image_name:
            image_name = f"{REPO_ROOT.name.lower()}-{service}"

        # Check if image exists locally
        img_cmd = ["docker", "images", "-q", image_name]
        img_res = subprocess.run(img_cmd, capture_output=True, text=True)

        if not img_res.stdout.strip():
            print(f">>> [Environment Missing] Automatically building '{image_name}' (this may take a few minutes)...")
            subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "build", service], cwd=REPO_ROOT, check=True)

    except Exception as e:
        print(f">>> [Warning] Image detection failed: {e}")


def native_time_stretch(input_path: Path, output_path: Path, rate: float) -> None:
    """
    Pre-calculates and saves time-stretched audio using native Python (librosa).
    Skips processing if the output file already exists.
    """
    if output_path.exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Load original audio
            y, sr = librosa.load(input_path, sr=None)
            # Apply time stretching
            y_stretched = librosa.effects.time_stretch(y, rate=rate)
            # Save to disk
            sf.write(output_path, y_stretched, sr)
    except Exception as e:
        raise RuntimeError(f"Audio time-stretch failed [{input_path.name}]: {e}")


def preprocess_all_audio(selected_pairs: list[SelectedWorkPair], ts_rates: list[float], aug_dir: Path) -> dict[
    float, dict[str, Path]]:
    """
    Phase 1: Pre-processes all query audio files on the host machine using MULTIPROCESSING.
    Generates time-stretched versions for specified rates.
    Returns a mapping: {time_stretch_rate: {original_path_str: augmented_path}}
    """
    print("\n>>> [Phase 1] Starting local audio pre-processing (Time Stretch) on host...")

    # Filter out rate 1.0 (no change needed)
    rates_to_process = [r for r in ts_rates if r != 1.0]

    if not rates_to_process:
        print(">>> No time-stretch processing required. Skipping.")
        return {1.0: {str(p.query_audio): p.query_audio for p in selected_pairs}}

    # Initialize mapping with rate 1.0 (original files)
    aug_mapping = {1.0: {str(p.query_audio): p.query_audio for p in selected_pairs}}

    # Set the number of worker processes to 8, matching the physical core count of the 5700X3D CPU for stability.
    # If system memory exceeds 32GB, you may increase max_workers to 14 to maximize throughput.
    MAX_WORKERS = 14

    for rate in rates_to_process:
        aug_mapping[rate] = {}
        rate_dir = aug_dir / f"ts_{rate}"
        rate_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- Generating {rate}x time-stretched audio (Multiprocessing) ---")

        # 1. Pre-build the task list and result mapping
        tasks = []
        for pair in selected_pairs:
            orig_path = pair.query_audio
            aug_path = rate_dir / f"{orig_path.stem}_ts{rate}.wav"
            aug_mapping[rate][str(orig_path)] = aug_path
            tasks.append((orig_path, aug_path, rate))

        # 2. Start the multiprocessing pool
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = {executor.submit(native_time_stretch, *task): task for task in tasks}

            # Track completion progress using tqdm
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Rate {rate}",
                               unit="file"):
                try:
                    future.result()  # Re-raise any exceptions that occurred in the subprocess
                except Exception as exc:
                    print(f"\n[Error] Task generated an exception: {exc}")
                    raise

    return aug_mapping


def run_embedding_extractor_docker(input_list: Path, model: str, output_dir: Path) -> None:
    """
    Phase 2: Runs the Docker container to extract embeddings using GPU.
    Supports 'clews' and 'discogs-vinet' models.
    """
    service_name = "clews" if model == "clews" else "discogs-vinet"
    ensure_docker_env(service_name)

    # Convert paths to container-compatible format
    input_rel = host_path_to_container(input_list).as_posix()
    output_rel = host_path_to_container(output_dir).as_posix()

    cmd = ["docker", "compose", "-f", COMPOSE_FILE, "run", "--rm"]

    if model == "clews":
        cmd.extend([
            "-e", "PYTHONPATH=/app:/app/models/clews",
            "clews", "python", "extractor/extractor_gpu_clew.py",
            "--input", input_rel,
            "--checkpoint", "models/clews/checkpoints/clews/dvi-clews/checkpoint_best.ckpt",
            "--output-path", output_rel
        ])
    elif model == "discogs-vinet":
        cmd.extend([
            "-e", "PYTHONPATH=/app:/app/models/Discogs-VINet",
            "discogs-vinet", "python", "extractor/extractor_vinet_gpu.py",
            "--input", input_rel,
            "--output-path", output_rel
        ])

    print(f"\n>>> [Phase 2] Launching GPU container for feature extraction ({model})...")
    process = subprocess.run(cmd, cwd=REPO_ROOT, check=False)

    if process.returncode != 0:
        raise RuntimeError(f"Docker extraction failed for {model}.")


def run_retrieval_evaluation_docker(first_list: Path, second_list: Path, labels_json: Path, model_name: str, dim: int,
                                    recall_ks: list[int], output_json: Path) -> None:
    """
    Phase 3: Runs the Docker container to evaluate retrieval performance (mAP, Recall).
    """
    ensure_docker_env("retrieval")

    cmd = [
        "docker", "compose", "-f", COMPOSE_FILE, "run", "--rm", "retrieval",
        "python", "retrieval/eval_retrieval.py",
        "--first-list", first_list.as_posix(),
        "--second-list", second_list.as_posix(),
        "--labels-json", labels_json.as_posix(),
        "--embedding-model", model_name,
        "--k", *[str(k) for k in recall_ks],
        "--output-json", output_json.as_posix(),
    ]

    print(f"\n>>> [Phase 3] Calculating mAP and Recall metrics for model [{model_name}]...")
    process = subprocess.run(cmd, cwd=REPO_ROOT, check=False)

    if process.returncode != 0:
        raise RuntimeError("Docker retrieval evaluation failed.")


def embedding_path_for_audio(audio_path: Path, embeddings_dir: Path, model: str) -> Path:
    """Determines the expected output path for an embedding file based on the model type."""
    suffix = ".pt" if model == "clews" else ".npy"
    return embeddings_dir / f"{audio_path.stem}{suffix}"


def run_single_experiment(config: dict, model_cfg: dict, ts_rate: float, selected_pairs: list[SelectedWorkPair],
                          aug_mapping: dict) -> dict[str, Any]:
    """
    Executes a single experiment iteration for a specific model and time-stretch rate.
    Handles data preparation, embedding extraction (if needed), and evaluation.
    """
    model_name = model_cfg["name"]
    dim = model_cfg["dim"]

    # Define experiment directories
    exp_dir = REPO_ROOT / config["output_base_dir"] / config["experiment_name"] / model_name / f"ts_{ts_rate}"
    embeddings_dir = exp_dir / "embeddings"
    exp_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Runtime directory for Docker volume mounting
    docker_runtime_root = REPO_ROOT / "extractor" / "pipeline_runtime" / f"{model_name}_ts{ts_rate}"
    docker_runtime_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Starting Experiment: Model=[{model_name}], Time-Stretch Rate=[{ts_rate}x]")
    print(f"{'=' * 50}")

    index_items, query_items = [], []

    for pair in selected_pairs:
        # Index always uses the original unmodified audio
        index_items.append(QueryItem(pair.index_audio, pair.work_id, f"{pair.work_id}::index", "index_original"))

        # Query uses the pre-processed time-stretched version
        mapped_query_path = aug_mapping[ts_rate][str(pair.query_audio)]
        source_tag = "query_original" if ts_rate == 1.0 else f"query_ts{ts_rate}"
        query_items.append(QueryItem(mapped_query_path, pair.work_id, f"{pair.work_id}::query", source_tag))

    all_audio_paths = [item.audio_path for item in index_items + query_items]
    ensure_unique_stems(all_audio_paths)

    # Prepare list of audio paths for Docker
    audio_list_path = docker_runtime_root / "all_audio_paths.txt"
    expected_embeddings = [embedding_path_for_audio(p, embeddings_dir, model_name) for p in all_audio_paths]

    # Identify missing embeddings
    missing_audio_paths = [p for p, emb in zip(all_audio_paths, expected_embeddings) if not emb.is_file()]

    if config.get("skip_embedding_extraction", False):
        print("\n>>> Skipping feature extraction (--skip-embedding-extraction=true).")
    else:
        if missing_audio_paths:
            # Convert missing paths to container format
            container_missing_paths = [host_path_to_container(p) for p in missing_audio_paths]
            write_path_list(container_missing_paths, audio_list_path)
            print(f"\n>>> Found {len(missing_audio_paths)} missing embeddings. Preparing to extract...")
            run_embedding_extractor_docker(audio_list_path, model_name, embeddings_dir)
        else:
            print("\n>>> All audio embeddings already exist. Skipping GPU extraction.")

    # Generate lists of embedding paths for evaluation
    first_list = exp_dir / "first_embeddings.txt"
    second_list = exp_dir / "second_embeddings.txt"

    write_path_list([embedding_path_for_audio(item.audio_path, embeddings_dir, model_name) for item in index_items],
                    first_list)
    write_path_list([embedding_path_for_audio(item.audio_path, embeddings_dir, model_name) for item in query_items],
                    second_list)

    # Generate Docker-compatible lists
    first_list_docker = docker_runtime_root / "first_embeddings_docker.txt"
    second_list_docker = docker_runtime_root / "second_embeddings_docker.txt"

    write_path_list([host_path_to_container(p) for p in
                     [embedding_path_for_audio(item.audio_path, embeddings_dir, model_name) for item in index_items]],
                    first_list_docker)
    write_path_list([host_path_to_container(p) for p in
                     [embedding_path_for_audio(item.audio_path, embeddings_dir, model_name) for item in query_items]],
                    second_list_docker)

    # Generate labels JSON
    labels_json = exp_dir / "embedding_labels.json"
    with labels_json.open("w", encoding="utf-8") as f:
        json.dump({item.audio_path.stem: {"work_id": item.work_id, "song_id": item.song_id} for item in
                   index_items + query_items}, f, indent=2)

    # Run evaluation
    eval_results_json = exp_dir / "eval_results.json"
    run_retrieval_evaluation_docker(
        host_path_to_container(first_list_docker),
        host_path_to_container(second_list_docker),
        host_path_to_container(labels_json),
        model_name,
        dim,
        config["recall_ks"],
        host_path_to_container(eval_results_json)
    )

    # Load and format results
    with eval_results_json.open("r", encoding="utf-8") as f:
        eval_output = json.load(f)

    report_data = {
        "model": model_name,
        "time_stretch_rate": ts_rate,
        "works_used": len(selected_pairs),
        "metrics": eval_output["metrics"]
    }

    # Save individual report
    with open(exp_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    return report_data


def main():
    """Main entry point for the orchestration pipeline."""
    config_path = REPO_ROOT / CONFIG_FILE

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file missing! Please create {CONFIG_FILE} in the root directory.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("==========================================")
    print("Launching Automated Ablation Study Orchestrator")
    print(f"Mode: [Native Python Pre-processing] + [Docker Feature Extraction/Evaluation]")
    print(f"Project Name: {config['experiment_name']}")
    print(f"Models to Test: {[m['name'] for m in config['models']]}")
    print(f"Time-Stretch Rates: {config['time_stretch_rates']}")
    print("==========================================")

    # Step 1: Parse data and select pairs
    work_to_paths = load_work_to_paths(REPO_ROOT / config["input_json"])
    selected_pairs = select_pairs(work_to_paths=work_to_paths, seed=config["seed"])

    # Step 2: Pre-generate all required augmented audio locally
    augmented_audio_dir = REPO_ROOT / config["output_base_dir"] / "augmented_audio"
    aug_mapping = preprocess_all_audio(selected_pairs, config["time_stretch_rates"], augmented_audio_dir)

    all_reports = []

    # Step 3: Run experiments (Nested loop over models and time-stretch rates)
    for model_cfg in config["models"]:
        for ts_rate in config["time_stretch_rates"]:
            try:
                report = run_single_experiment(config, model_cfg, ts_rate, selected_pairs, aug_mapping)
                all_reports.append(report)

                print(f"\n[Result] Model: {model_cfg['name']} | Time-Stretch: {ts_rate}x")
                print(f"   mAP: {report['metrics']['mAP']:.4f} | MR1: {report['metrics']['MR1']:.4f}")

            except Exception as e:
                print(f"\nExperiment Failed (Model={model_cfg['name']}, Rate={ts_rate}): {e}")

    # Save final summary
    summary_path = REPO_ROOT / config["output_base_dir"] / config["experiment_name"] / "final_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=4)

    print(f"\nAll experiments completed! Final report saved to:\n{summary_path.absolute()}")


if __name__ == "__main__":
    main()
"""
End-to-end Automated Cover Retrieval Pipeline Orchestrator.
Native Python Audio Pre-processing + Dockerized GPU Extraction & Evaluation.
[Pitch-Shift Ablation Only Version]
"""

from __future__ import annotations
import concurrent.futures
import json
import random
import subprocess
import warnings
import sys
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Configuration: Directories and files
DATASET_HOST_ROOT = Path(r"E:\Master\Master_Util\datasets_cover\discogs")
DATASET_CONTAINER_ROOT = Path("/data/discogs")
REPO_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = "docker-compose.windows-gpu.yml"
CONFIG_FILE = "config_final_PS_extreme.json"


class DualLogger:
    """
    Handles logging to both the terminal (stdout) and a log file simultaneously.
    Records the experiment start time upon initialization.
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.write(f"\n{'=' * 50}\n[Experiment Started At: {start_time}]\n{'=' * 50}\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


@dataclass
class SelectedWorkPair:
    """Represents a pair of audio files (index and query) belonging to the same musical work."""
    work_id: str
    index_audio: Path
    query_audio: Path


@dataclass
class QueryItem:
    """Represents a single audio item used for indexing or querying during evaluation."""
    audio_path: Path
    work_id: str
    song_id: str
    source: str


def load_work_to_paths(json_path: Path) -> dict[str, list[Path]]:
    """
    Loads a JSON mapping of work IDs to lists of audio file paths.
    Filters out invalid entries and ensures unique paths per work.
    Only returns works that have at least 2 associated audio files.
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

        if len(paths) >= 2:
            work_to_paths[work_id] = paths
    return work_to_paths


def _extract_path_from_item(item: Any) -> Path | None:
    """
    Extracts a valid file path from various input formats (string or dictionary).
    Resolves relative paths based on dataset root or repository root.
    """
    raw_val = None
    if isinstance(item, str):
        raw_val = item.strip()
    elif isinstance(item, dict):
        # Check common keys for path storage
        for key in ("path", "audio_path", "recording_path", "file_path", "filepath"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                raw_val = val.strip()
                break

    if not raw_val:
        return None

    # Handle container paths by mapping them back to the host filesystem
    if raw_val.startswith("/data/discogs"):
        rel_path = raw_val.replace("/data/discogs/", "").replace("/data/discogs", "").lstrip("/")
        return (DATASET_HOST_ROOT / rel_path).resolve()

    if raw_val.startswith("/"):
        return Path(raw_val)

    # Assume relative to repo root if no absolute prefix
    return (REPO_ROOT / raw_val).resolve()


def host_path_to_container(path: Path) -> Path:
    """
    Converts a host machine file path to the corresponding path inside the Docker container.
    Handles mappings for the repo root (/app) and dataset root (/data/discogs).
    """
    resolved_path = path.resolve()
    try:
        rel = resolved_path.relative_to(REPO_ROOT.resolve())
        return Path("/app") / rel.as_posix()
    except ValueError:
        pass
    try:
        rel = resolved_path.relative_to(DATASET_HOST_ROOT.resolve())
        return Path("/data/discogs") / rel.as_posix()
    except ValueError:
        pass
    # Fallback: simple slash conversion for Windows paths
    return Path(str(resolved_path).replace('\\', '/'))


def select_pairs(work_to_paths: dict[str, list[Path]], seed: int) -> list[SelectedWorkPair]:
    """
    Randomly selects one index audio and one query audio for each work ID.
    Uses a fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    selected: list[SelectedWorkPair] = []
    for work_id in sorted(work_to_paths.keys()):
        idx_audio, qry_audio = rng.sample(work_to_paths[work_id], 2)
        selected.append(SelectedWorkPair(work_id, idx_audio, qry_audio))
    return selected


def ensure_unique_stems(paths: list[Path]) -> None:
    """
    Verifies that all provided file paths have unique stems (filenames without extension).
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
    """Writes a list of file paths to a text file, one path per line."""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path.as_posix()}\n")


def ensure_docker_env(service: str) -> None:
    """
    Checks if the required Docker image exists. If not, automatically builds it.
    """
    try:
        cfg_cmd = ["docker", "compose", "-f", COMPOSE_FILE, "config", "--format", "json"]
        cfg_res = subprocess.run(cfg_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        config_data = json.loads(cfg_res.stdout)

        image_name = config_data.get("services", {}).get(service, {}).get("image")
        if not image_name:
            image_name = f"{REPO_ROOT.name.lower()}-{service}"

        img_cmd = ["docker", "images", "-q", image_name]
        img_res = subprocess.run(img_cmd, capture_output=True, text=True)

        if not img_res.stdout.strip():
            print(f">>> [Environment Missing] Automatically building '{image_name}'...")
            subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "build", service], cwd=REPO_ROOT, check=True)
    except Exception as e:
        print(f">>> [Warning] Image detection failed: {e}")


def process_single_audio_pitch_shift(input_path: Path, steps_and_outputs: list[tuple[int, Path]]) -> None:
    """
    Optimized Processing: Loads audio once, applies Pitch-Shift variations in memory, and saves results.
    Skips processing if output files already exist.
    """
    tasks_to_do = [(step, out_path) for step, out_path in steps_and_outputs if not out_path.exists()]

    if not tasks_to_do:
        return

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Load audio only once for efficiency
            y, sr = librosa.load(input_path, sr=None)

            for step, out_path in tasks_to_do:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=step)
                sf.write(out_path, y_shifted, sr)

    except Exception as e:
        raise RuntimeError(f"Audio processing failed [{input_path.name}]: {e}")


def preprocess_all_audio(selected_pairs: list[SelectedWorkPair], ps_steps: list[int], aug_dir: Path) -> dict:
    """
    Generates pitch-shifted versions of all query audios in parallel.
    Returns a mapping of {pitch_step: {original_path: shifted_path}}.
    Step 0 represents the original audio.
    """
    print("\n>>> [Phase 1] Starting fast preprocessing (Pitch-Shift Only)...")

    steps_to_process = [s for s in ps_steps if s != 0]
    aug_mapping = {0: {str(p.query_audio): p.query_audio for p in selected_pairs}}

    if not steps_to_process:
        print(">>> No pitch-shift processing required. Skipping.")
        return aug_mapping

    for step in steps_to_process:
        aug_mapping[step] = {}
        (aug_dir / f"ps_{step}").mkdir(parents=True, exist_ok=True)

    file_tasks = []

    for pair in selected_pairs:
        orig_path = pair.query_audio
        steps_and_outputs = []
        for step in steps_to_process:
            aug_path = aug_dir / f"ps_{step}" / f"{orig_path.stem}_ps{step}.wav"
            aug_mapping[step][str(orig_path)] = aug_path
            steps_and_outputs.append((step, aug_path))

        file_tasks.append((orig_path, steps_and_outputs))

    MAX_WORKERS = 14
    print(f"--- Processing {len(file_tasks)} songs using {MAX_WORKERS} concurrent processes ---")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_audio_pitch_shift, orig_path, so): orig_path for orig_path, so in
                   file_tasks}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Applying Pitch-Shift",
                           unit="song"):
            try:
                future.result()
            except Exception as exc:
                print(f"\n[Skipped File] Corrupt data or processing error: {exc}")

    return aug_mapping


def run_embedding_extractor_docker(input_list: Path, model: str, output_dir: Path) -> None:
    """
    Launches a Docker container to extract audio embeddings using the specified model (CLEWS or Discogs-VINet).
    """
    service_name = "clews" if model == "clews" else "discogs-vinet"
    ensure_docker_env(service_name)

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
    Launches a Docker container to perform retrieval evaluation (mAP, Recall@K).
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
    """Determines the expected file path for an embedding based on the audio path and model type."""
    suffix = ".pt" if model == "clews" else ".npy"
    return embeddings_dir / f"{audio_path.stem}{suffix}"


def run_single_experiment(config: dict, model_cfg: dict, ps_step: int, selected_pairs: list[SelectedWorkPair],
                          aug_mapping: dict) -> dict[str, Any]:
    """
    Executes a single experiment iteration for a specific model and pitch-shift step.
    Handles embedding extraction (if missing) and retrieval evaluation.
    """
    model_name = model_cfg["name"]
    dim = model_cfg["dim"]

    exp_dir = REPO_ROOT / config["output_base_dir"] / config["experiment_name"] / model_name / f"ps_{ps_step}"
    embeddings_dir = exp_dir / "embeddings"
    exp_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    docker_runtime_root = REPO_ROOT / "extractor" / "pipeline_runtime" / f"{model_name}_ps{ps_step}"
    docker_runtime_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Starting Experiment: Model=[{model_name}], Pitch-Shift=[{ps_step} semitones]")
    print(f"{'=' * 50}")

    index_items, query_items = [], []

    for pair in selected_pairs:
        # Index always uses the original audio
        index_items.append(QueryItem(pair.index_audio, pair.work_id, f"{pair.work_id}::index", "index_original"))
        # Query uses the pitch-shifted version corresponding to the current step
        mapped_query_path = aug_mapping[ps_step][str(pair.query_audio)]
        source_tag = "query_original" if ps_step == 0 else f"query_ps{ps_step}"
        query_items.append(QueryItem(mapped_query_path, pair.work_id, f"{pair.work_id}::query", source_tag))

    all_audio_paths = [item.audio_path for item in index_items + query_items]
    ensure_unique_stems(all_audio_paths)

    audio_list_path = docker_runtime_root / "all_audio_paths.txt"
    expected_embeddings = [embedding_path_for_audio(p, embeddings_dir, model_name) for p in all_audio_paths]

    missing_audio_paths = [p for p, emb in zip(all_audio_paths, expected_embeddings) if not emb.is_file()]

    if config.get("skip_embedding_extraction", False):
        print("\n>>> Skipping feature extraction (--skip-embedding-extraction=true).")
    else:
        if missing_audio_paths:
            container_missing_paths = [host_path_to_container(p) for p in missing_audio_paths]
            write_path_list(container_missing_paths, audio_list_path)
            print(f"\n>>> Found {len(missing_audio_paths)} missing embeddings. Preparing to extract...")
            run_embedding_extractor_docker(audio_list_path, model_name, embeddings_dir)
        else:
            print("\n>>> All audio embeddings already exist. Skipping GPU extraction.")

    # Filter out pairs where embeddings are missing
    valid_index_items = []
    valid_query_items = []

    for idx_item, qry_item in zip(index_items, query_items):
        idx_emb_path = embedding_path_for_audio(idx_item.audio_path, embeddings_dir, model_name)
        qry_emb_path = embedding_path_for_audio(qry_item.audio_path, embeddings_dir, model_name)

        if idx_emb_path.exists() and qry_emb_path.exists():
            valid_index_items.append(idx_item)
            valid_query_items.append(qry_item)
        else:
            print(f"[Skip Evaluation] Features missing, removed from evaluation list: {idx_item.work_id}")

    # Generate text files listing embedding paths for the Docker container
    first_list = exp_dir / "first_embeddings.txt"
    second_list = exp_dir / "second_embeddings.txt"
    write_path_list(
        [embedding_path_for_audio(item.audio_path, embeddings_dir, model_name) for item in valid_index_items],
        first_list)
    write_path_list(
        [embedding_path_for_audio(item.audio_path, embeddings_dir, model_name) for item in valid_query_items],
        second_list)

    first_list_docker = docker_runtime_root / "first_embeddings_docker.txt"
    second_list_docker = docker_runtime_root / "second_embeddings_docker.txt"
    write_path_list(
        [host_path_to_container(embedding_path_for_audio(item.audio_path, embeddings_dir, model_name)) for item in
         valid_index_items], first_list_docker)
    write_path_list(
        [host_path_to_container(embedding_path_for_audio(item.audio_path, embeddings_dir, model_name)) for item in
         valid_query_items], second_list_docker)

    # Generate JSON labels mapping file stems to work/song IDs
    labels_json = exp_dir / "embedding_labels.json"
    with labels_json.open("w", encoding="utf-8") as f:
        valid_labels = {
            item.audio_path.stem: {"work_id": item.work_id, "song_id": item.song_id}
            for item in valid_index_items + valid_query_items
        }
        json.dump(valid_labels, f, indent=2)

    print(
        f"   -> Evaluation list filtering complete: Original {len(index_items)} pairs, {len(valid_index_items)} valid pairs retained.")

    # Run evaluation inside Docker
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

    with eval_results_json.open("r", encoding="utf-8") as f:
        eval_output = json.load(f)

    report_data = {
        "model": model_name,
        "pitch_shift_step": ps_step,
        "works_used": len(selected_pairs),
        "metrics": eval_output["metrics"]
    }

    with open(exp_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    return report_data


def main():
    """Main entry point for the orchestration pipeline."""
    sys.stdout = DualLogger("ablation_pitch_shift_final.log")

    config_path = REPO_ROOT / CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file missing! Please create {CONFIG_FILE} in the root directory.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ps_steps = config.get("pitch_shift_steps", [0])

    print("==========================================")
    print("Launching Automated Ablation Study Orchestrator")
    print(f"Mode: [Pitch-Shift Only] + [Docker Feature Extraction/Evaluation]")
    print(f"Project Name: {config['experiment_name']}")
    print(f"Models to Test: {[m['name'] for m in config['models']]}")
    print(f"Pitch-Shift Steps: {ps_steps}")
    print("==========================================")

    work_to_paths = load_work_to_paths(REPO_ROOT / config["input_json"])
    selected_pairs = select_pairs(work_to_paths=work_to_paths, seed=config["seed"])

    augmented_audio_dir = REPO_ROOT / config["output_base_dir"] / "augmented_audio"

    # Preprocess all pitch-shifted audio files
    aug_mapping = preprocess_all_audio(selected_pairs, ps_steps, augmented_audio_dir)

    all_reports = []

    # Nested loop: Iterate over each model and each pitch-shift step
    for model_cfg in config["models"]:
        for ps_step in ps_steps:
            try:
                report = run_single_experiment(config, model_cfg, ps_step, selected_pairs, aug_mapping)
                all_reports.append(report)

                print(f"\n[Result] Model: {model_cfg['name']} | Pitch-Shift: {ps_step} steps")
                print(f"   mAP: {report['metrics']['mAP']:.4f} | MR1: {report['metrics']['MR1']:.4f}")

            except Exception as e:
                print(f"\nExperiment Failed (Model={model_cfg['name']}, PS={ps_step}): {e}")

    summary_path = REPO_ROOT / config["output_base_dir"] / config["experiment_name"] / "final_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=4)

    print(f"\nAll experiments completed! Final report saved to:\n{summary_path.absolute()}")


if __name__ == "__main__":
    main()
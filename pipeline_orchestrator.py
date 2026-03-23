"""End-to-end cover retrieval pipeline orchestrator.

This script wires together the repository components in one flow:
1) Read input JSON mapping work IDs to recording paths (or metadata objects).
2) Randomly select one recording per work for index and another for query.
3) Compute embeddings for all selected audio via extractor/extractor.py.
4) Evaluate retrieval in the retrieval docker service.

Outputs:
- Human-readable metrics in stdout.
- Structured JSON report with selections and evaluation metrics.
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
DEGRADATION_CONTAINER_REPO_ROOT = Path("/app")
DEGRADATION_DISCOGS_HOST_ROOT = Path("/Volumes/T7 Shield/discogs")
DEGRADATION_DISCOGS_CONTAINER_ROOT = Path("/data/discogs")


@dataclass
class SelectedWorkPair:
    work_id: str
    index_audio: Path
    query_audio: Path


@dataclass
class QueryItem:
    audio_path: Path
    work_id: str
    song_id: str
    source: str


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_dim: int
    metric: str
    normalize: bool


@dataclass(frozen=True)
class AugmentationResult:
    output_path: Path
    kind: str
    params: dict[str, Any]


def _normalize_path(raw: str) -> Path:
    """Resolve repo-relative paths while preserving absolute paths unchanged."""
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _extract_path_from_item(
    item: Any,
) -> Path | None:
    """Accept either a plain string path or a metadata dict with a known path field."""
    if isinstance(item, str):
        return _normalize_path(item)

    if not isinstance(item, dict):
        return None

    for key in ("path", "audio_path", "recording_path", "file_path", "filepath"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_path(value.strip())

    return None


def load_work_to_paths(
    json_path: Path,
    require_existing_files: bool,
) -> dict[str, list[Path]]:
    """Load the input JSON and keep only works that provide at least two distinct paths."""
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object: {work_id: [recordings...]}")

    work_to_paths: dict[str, list[Path]] = {}
    for raw_work_id, items in data.items():
        work_id = str(raw_work_id)
        if not isinstance(items, list):
            continue

        seen: set[Path] = set()
        paths: list[Path] = []
        for item in items:
            candidate = _extract_path_from_item(item)
            if candidate is None:
                continue
            if require_existing_files and (not candidate.is_file()):
                continue
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)

        if len(paths) >= 2:
            work_to_paths[work_id] = paths

    return work_to_paths


def select_pairs(
    work_to_paths: dict[str, list[Path]], seed: int
) -> list[SelectedWorkPair]:
    """Pick one index recording and one query recording per work in a reproducible way."""
    rng = random.Random(seed)
    selected: list[SelectedWorkPair] = []

    for work_id in sorted(work_to_paths.keys()):
        candidates = work_to_paths[work_id]
        idx_audio, qry_audio = rng.sample(candidates, 2)
        selected.append(
            SelectedWorkPair(
                work_id=work_id,
                index_audio=Path(idx_audio),
                query_audio=Path(qry_audio),
            )
        )

    return selected


def ensure_unique_stems(paths: list[Path]) -> None:
    """Fail fast when two different files would map to the same embedding basename."""
    stems: dict[str, Path] = {}
    duplicates: list[tuple[str, Path, Path]] = []

    for path in paths:
        stem = path.stem
        if stem in stems and stems[stem] != path:
            duplicates.append((stem, stems[stem], path))
        else:
            stems[stem] = path

    if duplicates:
        lines = [
            "Found duplicate file stems. Embedding IDs are stem-based and would collide:"
        ]
        for stem, p1, p2 in duplicates[:10]:
            lines.append(f"  stem='{stem}' -> '{p1}' and '{p2}'")
        if len(duplicates) > 10:
            lines.append(f"  ... plus {len(duplicates) - 10} more")
        raise ValueError("\n".join(lines))


def write_path_list(paths: list[Path], output_txt: Path) -> None:
    """Write one filesystem path per line for downstream CLI tools."""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path}\n")


def extractor_host_path_to_container(path: Path) -> Path:
    """Translate a host path under extractor/ into the path seen inside Docker containers."""
    extractor_host = (REPO_ROOT / "extractor").resolve()
    extractor_container = Path("/app/extractor")

    try:
        rel = path.resolve().relative_to(extractor_host)
    except ValueError as exc:
        raise ValueError(f"Path must be inside extractor/: {path}") from exc

    return extractor_container / rel


def _docker_service_from_model(model: str) -> str:
    """Map the embedding model name to the docker-compose service name."""
    if model == "clews":
        return "clews"
    if model == "discogs-vinet":
        return "discogs-vinet"
    raise ValueError(f"Unsupported embedding model: {model}")


def retrieval_config_from_model(model: str) -> RetrievalConfig:
    """Return retrieval settings that must match the selected embedding model."""
    if model == "clews":
        return RetrievalConfig(embedding_dim=1024, metric="l2", normalize=False)
    if model == "discogs-vinet":
        return RetrievalConfig(embedding_dim=512, metric="ip", normalize=True)
    raise ValueError(f"Unsupported embedding model: {model}")


def _ensure_path_under(path: Path, base: Path) -> None:
    """Guard against passing paths outside the subset mounted into containers."""
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be inside '{base}': {path}") from exc


def _path_for_extractor_container(path: Path) -> Path:
    """Map extractor-local files to /app/extractor; keep externally mounted paths as-is."""
    extractor_host = (REPO_ROOT / "extractor").resolve()
    try:
        path.resolve().relative_to(extractor_host)
    except ValueError:
        return path
    return extractor_host_path_to_container(path)


def _path_for_degradation_container(path: Path) -> Path:
    """Map host paths to the degradation container mount layout."""
    resolved = path.resolve()

    try:
        rel_repo = resolved.relative_to(REPO_ROOT.resolve())
        return DEGRADATION_CONTAINER_REPO_ROOT / rel_repo
    except ValueError:
        pass

    try:
        rel_discogs = resolved.relative_to(DEGRADATION_DISCOGS_HOST_ROOT.resolve())
        return DEGRADATION_DISCOGS_CONTAINER_ROOT / rel_discogs
    except ValueError:
        pass

    return resolved


def _format_pitch_shift_token(n_steps: float) -> str:
    token = str(n_steps).replace(".", "p").replace("-", "m")
    return token


def _format_time_stretch_token(rate: float) -> str:
    token = str(rate).replace(".", "p").replace("-", "m")
    return token


def apply_pitch_shift_augmentation(
    input_audio: Path,
    output_audio: Path,
    n_steps: float,
) -> None:
    """Create a pitch-shifted copy using the degradation Docker service."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    input_for_container = _path_for_degradation_container(input_audio)
    output_for_container = _path_for_degradation_container(output_audio)

    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "degradation",
        "python",
        "degradation/pitch_shift.py",
        "--input",
        str(input_for_container),
        "--output",
        str(output_for_container),
        "--n-steps",
        str(n_steps),
    ]
    print("Running pitch-shift augmentation docker command:")
    print(" ".join(cmd))

    process = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Pitch-shift augmentation failed in the degradation docker service."
        )


def apply_time_stretch_augmentation(
    input_audio: Path,
    output_audio: Path,
    stretch_rate: float,
) -> None:
    """Create a time-stretched copy using the degradation Docker service."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    input_for_container = _path_for_degradation_container(input_audio)
    output_for_container = _path_for_degradation_container(output_audio)

    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "degradation",
        "python",
        "degradation/time_stretch.py",
        "--input",
        str(input_for_container),
        "--output",
        str(output_for_container),
        "--stretch-rate",
        str(stretch_rate),
        "--backend",
        "librosa",
    ]
    print("Running time-stretch augmentation docker command:")
    print(" ".join(cmd))

    process = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Time-stretch augmentation failed in the degradation docker service."
        )


def run_embedding_extractor_docker(
    input_list: Path,
    model: str,
    output_dir: Path,
    docker_build_first: bool,
) -> None:
    """Run the model-specific extractor inside its Docker service."""
    service = _docker_service_from_model(model)

    # In docker compose, only selected folders are mounted. Keep runtime files in extractor/.
    mounted_base = REPO_ROOT / "extractor"
    _ensure_path_under(input_list, mounted_base)
    _ensure_path_under(output_dir, mounted_base)

    # The container sees the repo mounted at /app, so pass repo-relative paths here.
    input_rel = input_list.resolve().relative_to(REPO_ROOT)
    output_rel = output_dir.resolve().relative_to(REPO_ROOT)

    if docker_build_first:
        build_cmd = ["docker", "compose", "build", service]
        print("Running docker build command:")
        print(" ".join(build_cmd))
        build_process = subprocess.run(build_cmd, cwd=REPO_ROOT, check=False)
        if build_process.returncode != 0:
            raise RuntimeError(f"Docker build failed for service '{service}'.")

    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        service,
        "python",
        "extractor/extractor.py",
        "--input",
        str(input_rel),
        "--model",
        model,
        "--output-path",
        str(output_rel),
    ]

    print("Running embedding extraction docker command:")
    print(" ".join(cmd))

    process = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Docker embedding extraction failed. Verify docker compose is running "
            "and model container dependencies are available."
        )


def run_retrieval_evaluation_docker(
    first_list: Path,
    second_list: Path,
    labels_json: Path,
    embedding_model: str,
    recall_ks: tuple[int, ...],
    output_json: Path,
    verbose: bool,
) -> None:
    """Run retrieval evaluation inside the retrieval service with explicit label mappings."""
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "retrieval",
        "python",
        "retrieval/eval_retrieval.py",
        "--first-list",
        str(first_list),
        "--second-list",
        str(second_list),
        "--embedding-model",
        str(embedding_model),
        "--labels-json",
        str(labels_json),
        "--k",
        *[str(k) for k in recall_ks],
        "--output-json",
        str(output_json),
    ]
    if verbose:
        cmd.append("--verbose")

    print("Running retrieval evaluation docker command:")
    print(" ".join(cmd))

    process = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError("Docker retrieval evaluation failed.")


def embedding_path_for_audio(
    audio_path: Path, embeddings_dir: Path, model: str
) -> Path:
    """Derive the embedding filename from the audio stem and model output format."""
    suffix = ".pt" if model == "clews" else ".npy"
    return embeddings_dir / f"{audio_path.stem}{suffix}"


def parse_int_list(raw: str) -> tuple[int, ...]:
    """Parse comma-separated integers from the CLI into a tuple."""
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected a non-empty comma-separated int list.")
    return tuple(values)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    retrieval_config = retrieval_config_from_model(args.embedding_model)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "report.json"

    # Runtime helper files live under extractor/ because that subtree is mounted into the
    # model and retrieval containers. The final report still lives in the requested output dir.
    docker_runtime_root = (
        REPO_ROOT / "extractor" / ".pipeline_runtime" / output_dir.name
    )
    docker_runtime_root.mkdir(parents=True, exist_ok=True)
    augmentation_root = docker_runtime_root / "augmented_queries"
    augmentation_root.mkdir(parents=True, exist_ok=True)

    work_to_paths = load_work_to_paths(
        json_path=args.input_json,
        require_existing_files=False,
    )
    if not work_to_paths:
        raise ValueError(
            "No valid works found with at least two usable paths. "
            "Check input JSON and file existence."
        )

    selected_pairs = select_pairs(work_to_paths=work_to_paths, seed=args.seed)

    index_items: list[QueryItem] = []
    for pair in selected_pairs:
        index_items.append(
            QueryItem(
                audio_path=pair.index_audio,
                work_id=pair.work_id,
                song_id=f"{pair.work_id}::index",
                source="index",
            )
        )

    query_items = [
        QueryItem(
            audio_path=pair.query_audio,
            work_id=pair.work_id,
            song_id=f"{pair.work_id}::query",
            source="original_query",
        )
        for pair in selected_pairs
    ]

    augmentation_records: list[AugmentationResult] = []
    if args.enable_pitch_shift_augmentation and args.enable_time_stretch_augmentation:
        raise ValueError(
            "Choose only one query augmentation mode: pitch shift or time stretch."
        )

    if args.enable_pitch_shift_augmentation:
        token = _format_pitch_shift_token(args.pitch_shift_n_steps)
        augmented_query_items: list[QueryItem] = []
        for item in query_items:
            out_name = f"{item.audio_path.stem}__pitch_shift_{token}.wav"
            out_path = augmentation_root / out_name
            apply_pitch_shift_augmentation(
                input_audio=item.audio_path,
                output_audio=out_path,
                n_steps=args.pitch_shift_n_steps,
            )
            augmentation_records.append(
                AugmentationResult(
                    output_path=out_path,
                    kind="pitch_shift",
                    params={"n_steps": args.pitch_shift_n_steps},
                )
            )
            augmented_query_items.append(
                QueryItem(
                    audio_path=out_path,
                    work_id=item.work_id,
                    song_id=f"{item.work_id}::query_pitch_shift",
                    source="pitch_shift_query",
                )
            )

        query_items = augmented_query_items

    if args.enable_time_stretch_augmentation:
        token = _format_time_stretch_token(args.time_stretch_rate)
        augmented_query_items: list[QueryItem] = []
        for item in query_items:
            out_name = f"{item.audio_path.stem}__time_stretch_{token}.wav"
            out_path = augmentation_root / out_name
            apply_time_stretch_augmentation(
                input_audio=item.audio_path,
                output_audio=out_path,
                stretch_rate=args.time_stretch_rate,
            )
            augmentation_records.append(
                AugmentationResult(
                    output_path=out_path,
                    kind="time_stretch",
                    params={"stretch_rate": args.time_stretch_rate},
                )
            )
            augmented_query_items.append(
                QueryItem(
                    audio_path=out_path,
                    work_id=item.work_id,
                    song_id=f"{item.work_id}::query_time_stretch",
                    source="time_stretch_query",
                )
            )

        query_items = augmented_query_items

    # Retrieval labels are keyed by embedding stem, so stem collisions would corrupt evaluation.
    all_audio_paths = [item.audio_path for item in index_items + query_items]
    ensure_unique_stems(all_audio_paths)

    audio_list_path = docker_runtime_root / "all_audio_paths.txt"
    expected_embeddings = [
        embedding_path_for_audio(path, embeddings_dir, args.embedding_model)
        for path in all_audio_paths
    ]
    missing_audio_paths = [
        path
        for path, emb_path in zip(all_audio_paths, expected_embeddings)
        if not emb_path.is_file()
    ]

    if args.skip_embedding_extraction:
        if missing_audio_paths:
            raise FileNotFoundError(
                "--skip-embedding-extraction was set, but some embeddings are missing. "
                f"Missing: {len(missing_audio_paths)}"
            )
        print(
            "Skipping embedding extraction as requested. All embeddings already exist."
        )
    else:
        if missing_audio_paths:
            # The extractor consumes audio paths exactly as listed here; mounting those
            # audio locations into the container is the caller's responsibility.
            write_path_list(
                [_path_for_extractor_container(p) for p in missing_audio_paths],
                audio_list_path,
            )
            print(
                "Embeddings already present for "
                f"{len(all_audio_paths) - len(missing_audio_paths)} files; extracting only "
                f"{len(missing_audio_paths)} missing files."
            )
            run_embedding_extractor_docker(
                input_list=audio_list_path,
                model=args.embedding_model,
                output_dir=embeddings_dir,
                docker_build_first=args.docker_build_first,
            )
        else:
            write_path_list([], audio_list_path)
            print("All expected embeddings already exist. Skipping extraction.")

    index_embeddings: list[Path] = []
    for item in index_items:
        emb_path = embedding_path_for_audio(
            item.audio_path, embeddings_dir, args.embedding_model
        )
        if not emb_path.is_file():
            raise FileNotFoundError(f"Missing index embedding file: {emb_path}")
        index_embeddings.append(emb_path)

    query_embeddings: list[Path] = []
    for item in query_items:
        emb_path = embedding_path_for_audio(
            item.audio_path, embeddings_dir, args.embedding_model
        )
        if not emb_path.is_file():
            raise FileNotFoundError(f"Missing query embedding file: {emb_path}")
        query_embeddings.append(emb_path)

    first_list = output_dir / "first_embeddings.txt"
    second_list = output_dir / "second_embeddings.txt"
    write_path_list(index_embeddings, first_list)
    write_path_list(query_embeddings, second_list)

    first_list_docker = docker_runtime_root / "first_embeddings_docker.txt"
    second_list_docker = docker_runtime_root / "second_embeddings_docker.txt"
    # The retrieval container cannot use host paths directly, so write a second pair of
    # list files using the mounted /app/extractor view of the same embedding files.
    write_path_list(
        [extractor_host_path_to_container(p) for p in index_embeddings],
        first_list_docker,
    )
    write_path_list(
        [extractor_host_path_to_container(p) for p in query_embeddings],
        second_list_docker,
    )

    stem_to_work_id: dict[str, str] = {}
    stem_to_song_id: dict[str, str] = {}
    for item in index_items + query_items:
        stem = item.audio_path.stem
        stem_to_work_id[stem] = item.work_id
        stem_to_song_id[stem] = item.song_id

    # eval_retrieval.py accepts an explicit stem -> {work_id, song_id} mapping, which avoids
    # depending on the original dataset metadata format during orchestration.
    labels_json = output_dir / "embedding_labels.json"
    with labels_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                stem: {
                    "work_id": stem_to_work_id[stem],
                    "song_id": stem_to_song_id[stem],
                }
                for stem in sorted(stem_to_work_id.keys())
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    eval_results_json = output_dir / "eval_results.json"
    run_retrieval_evaluation_docker(
        first_list=extractor_host_path_to_container(first_list_docker),
        second_list=extractor_host_path_to_container(second_list_docker),
        labels_json=extractor_host_path_to_container(labels_json),
        embedding_model=args.embedding_model,
        recall_ks=parse_int_list(args.recall_ks),
        output_json=extractor_host_path_to_container(eval_results_json),
        verbose=args.verbose_eval,
    )

    if not eval_results_json.is_file():
        raise FileNotFoundError(
            f"Expected evaluation output not found: {eval_results_json}"
        )
    with eval_results_json.open("r", encoding="utf-8") as f:
        eval_output = json.load(f)

    if "metrics" not in eval_output or "details" not in eval_output:
        raise ValueError("Invalid eval output JSON: missing 'metrics' or 'details'")

    metrics = eval_output["metrics"]
    details = eval_output["details"]

    report = {
        "input_json": str(args.input_json),
        "seed": args.seed,
        "works_total_in_json": len(work_to_paths),
        "works_used": len(selected_pairs),
        "augmentations_enabled": (
            args.enable_pitch_shift_augmentation or args.enable_time_stretch_augmentation
        ),
        "augmentation_summary": {
            "pitch_shift": sum(1 for rec in augmentation_records if rec.kind == "pitch_shift"),
            "time_stretch": sum(1 for rec in augmentation_records if rec.kind == "time_stretch"),
            "reverb": 0,
            "files": [
                {
                    "path": str(rec.output_path),
                    "type": rec.kind,
                    "params": rec.params,
                }
                for rec in augmentation_records
            ],
        },
        "embedding_model": args.embedding_model,
        "embedding_dim": retrieval_config.embedding_dim,
        "retrieval_metric": retrieval_config.metric,
        "retrieval_normalize": retrieval_config.normalize,
        "output_dir": str(output_dir),
        "embeddings_dir": str(embeddings_dir),
        "report_json": str(report_json),
        "lists": {
            "all_audio_paths": str(audio_list_path),
            "first_embeddings": str(first_list),
            "second_embeddings": str(second_list),
            "first_embeddings_docker": str(first_list_docker),
            "second_embeddings_docker": str(second_list_docker),
        },
        "labels_json": str(labels_json),
        "eval_results_json": str(eval_results_json),
        "selection": [
            {
                "work_id": pair.work_id,
                "index_audio": str(pair.index_audio),
                "query_audio": str(pair.query_audio),
            }
            for pair in selected_pairs
        ],
        "metrics": metrics,
        "details": details,
    }

    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the orchestrator entrypoint."""
    parser = argparse.ArgumentParser(
        description="General orchestrator for selection, optional augmentation, embeddings, and retrieval evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help=(
            "Input JSON mapping work_id to recording paths. "
            "Items must be path strings or dicts containing path/audio_path/recording_path/file_path/filepath."
        ),
    )
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="If set, do not check path existence while loading JSON.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for pair selection."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pipeline_runs/latest"),
        help="Output directory. The script writes report.json and embeddings/ inside it.",
    )

    parser.add_argument(
        "--skip-embedding-extraction",
        action="store_true",
        help="Skip extractor call and use existing embedding files from output-dir/embeddings.",
    )
    parser.add_argument(
        "--docker-build-first",
        action="store_true",
        help="Run 'docker compose build <service>' before extraction.",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["clews", "discogs-vinet"],
        required=True,
        help="Embedding model argument passed to extractor.",
    )
    parser.add_argument(
        "--enable-pitch-shift-augmentation",
        action="store_true",
        help="If set, replace query audio with pitch-shifted versions before embedding extraction.",
    )
    parser.add_argument(
        "--pitch-shift-n-steps",
        type=float,
        default=2.0,
        help="Semitone shift used when --enable-pitch-shift-augmentation is enabled.",
    )
    parser.add_argument(
        "--enable-time-stretch-augmentation",
        action="store_true",
        help="If set, replace query audio with time-stretched versions before embedding extraction.",
    )
    parser.add_argument(
        "--time-stretch-rate",
        type=float,
        default=1.2,
        help="Stretch rate used when --enable-time-stretch-augmentation is enabled.",
    )

    parser.add_argument(
        "--recall-ks",
        type=str,
        default="1,10,100",
        help="Comma-separated recall@K values.",
    )
    parser.add_argument(
        "--verbose-eval",
        action="store_true",
        help="Enable verbose output in retrieval/eval_retrieval.py.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, run the pipeline, and persist the final report."""
    parser = build_parser()
    args = parser.parse_args()

    args.input_json = args.input_json.resolve()
    args.output_dir = args.output_dir.resolve()

    embeddings_dir = args.output_dir / "embeddings"
    _ensure_path_under(embeddings_dir, REPO_ROOT / "extractor")

    report = run_pipeline(args)

    report_path = args.output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print("\nPipeline completed.")
    print(f"Works used: {report['works_used']}")
    print(f"mAP:  {report['metrics']['mAP']:.6f}")
    print(f"MR1:  {report['metrics']['MR1']:.6f}")
    print(f"NAR:  {report['metrics']['NAR']:.6f}")

    recall_keys = sorted([k for k in report["metrics"].keys() if k.startswith("R@")])
    for key in recall_keys:
        print(f"{key}: {report['metrics'][key]:.6f}")

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

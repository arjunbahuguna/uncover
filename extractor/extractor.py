"""
Audio Embedding Extractor Module

This module extracts audio embeddings for musical version matching using either
CLEWS or Discogs-VINet models. It processes audio files provided in a txt file
and generates embeddings using the specified model.

The code is designed to run within Docker containers, automatically detecting
and using the appropriate model based on the environment.
"""

import sys
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_embeddings_clews(file_paths: list, output_path: Path) -> None:
    """
    Extract embeddings using the CLEWS model.

    Args:
        file_paths: List of Path objects to audio files
        output_path: Path to output directory for embeddings
    """

    # Process each file individually
    for file_path in file_paths:
        output_file = output_path / f"{file_path.stem}.pt"

        if output_file.exists():
            logger.info(f"Skipping {file_path}: output already exists at {output_file}")
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            temp_wav_path = tmp_dir_path / f"{file_path.stem}.wav"

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(file_path),
                str(temp_wav_path),
            ]
            ffmpeg_result = subprocess.run(
                ffmpeg_cmd, cwd="/app", capture_output=True, text=True
            )
            if ffmpeg_result.returncode != 0:
                logger.error(
                    f"Failed to convert {file_path} to temporary wav: {ffmpeg_result.stderr}"
                )
                sys.exit(1)

            cmd = [
                "python",
                "inference.py",
                "--checkpoint=checkpoints/clews/dvi-clews/checkpoint_best.ckpt",
                f"--fn_in={str(temp_wav_path)}",
                f"--fn_out={str(output_file)}",
                "--device=cpu",
            ]

            logger.info(f"Processing {file_path} -> {output_file}")
            result = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"CLEWS inference failed for {file_path}: {result.stderr}")
                sys.exit(1)

        logger.info(f"Successfully extracted embedding for {file_path}")

    logger.info("CLEWS embedding extraction completed.")


def extract_embeddings_discogs_vinet(
    file_paths: list,
    output_path: Path,
    granularity: str = "track",
    fp16: bool = False,
) -> None:
    """
    Extract embeddings using the Discogs-VINet model.

    Args:
        file_paths: List of Path objects to audio files
        output_path: Path to output directory for embeddings
        granularity: Embedding granularity level ('track' or 'chunk')
        fp16: Store embeddings with FP16 precision
    """

    # Process each file individually
    for file_path in file_paths:
        output_file = output_path / f"{file_path.stem}.npy"

        if output_file.exists():
            logger.info(f"Skipping {file_path}: output already exists at {output_file}")
            continue

        temp_wav_path = None

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                temp_wav_path = tmp_dir_path / f"{file_path.stem}.wav"

                ffmpeg_cmd = [
                    "conda",
                    "run",
                    "-n",
                    "discogs-vinet",
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(file_path),
                    str(temp_wav_path),
                ]
                ffmpeg_result = subprocess.run(
                    ffmpeg_cmd, cwd="/app", capture_output=True, text=True
                )
                if ffmpeg_result.returncode != 0:
                    logger.error(
                        f"Failed to convert {file_path} to wav with ffmpeg: {ffmpeg_result.stderr}"
                    )
                    sys.exit(1)

                cmd = [
                    "conda",
                    "run",
                    "-n",
                    "discogs-vinet",
                    "python",
                    "inference.py",
                    str(tmp_dir_path),
                    "logs/checkpoints/Discogs-VINet-MIREX-full_set/config.yaml",
                    str(output_path),
                    f"--granularity={granularity}",
                ]

                if fp16:
                    cmd.append("--fp16")

                logger.info(f"Processing {file_path} -> {output_file}")
                result = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(
                        f"Discogs-VINet inference failed for {file_path}: {result.stderr}"
                    )
                    sys.exit(1)

                logger.info(f"Successfully extracted embedding for {file_path}")
        finally:
            if temp_wav_path is not None and temp_wav_path.exists():
                temp_wav_path.unlink()

    logger.info("Discogs-VINet embedding extraction completed.")


def main(args) -> None:
    """
    Main function to extract embeddings from audio files.

    Loads audio file paths from a txt file (one path per line), validates files exist,
    and runs the appropriate embedding extraction pipeline based on the model.

    Args:
        args: Command-line arguments containing:
            - input: Path to txt file with audio file paths (one per line)
            - model: Model to use ('clews' or 'discogs-vinet')
            - output_path: Output directory for embeddings
    """
    logger.info(f"Starting embedding extraction with model: {args.model}")

    # Read audio paths from txt file
    with open(args.input, "r") as f:
        path_list = [Path(line.strip()) for line in f if line.strip()]

    logger.info(f"Loaded {len(path_list)} audio paths from {args.input}")

    # Validate audio files exist
    valid_files = []
    for path in path_list:
        if not path.is_file():
            logger.warning(f"File {path} does not exist.")
        else:
            valid_files.append(path)

    if not valid_files:
        logger.error("No valid audio files found. Aborting extraction.")
        sys.exit(1)

    # Ensure output directory exists
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Processing {len(valid_files)} valid audio files")

    # Create a temporary input directory with symlinks for CLEWS
    if args.model.lower() == "clews":
        extract_embeddings_clews(valid_files, output_path)

    elif args.model.lower() == "discogs-vinet":
        extract_embeddings_discogs_vinet(valid_files, output_path)

    else:
        logger.error(
            f"Unknown model: {args.model}. Supported models: 'clews', 'discogs-vinet'"
        )
        sys.exit(1)

    logger.info("Embedding extraction pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings using CLEWS or Discogs-VINet models from audio files listed in a txt file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to txt file containing audio file paths (one path per line)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["clews", "discogs-vinet"],
        help="Model to use for embedding extraction",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory path for extracted embeddings",
    )
    args = parser.parse_args()

    main(args)

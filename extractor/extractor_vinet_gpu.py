import sys
import logging
import argparse
import subprocess
import tempfile
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Initialize logger
logger = logging.getLogger(__name__)


def convert_one_file(args):
    """
    Converts a single audio file to WAV format using FFmpeg.
    This function is designed to run in parallel CPU threads.
    """
    file_path, tmp_dir, model_sr, current_idx, total_count = args
    temp_wav_path = Path(tmp_dir) / f"{file_path.stem}.wav"

    # Print progress every 50 files or at the very end
    if current_idx % 50 == 0 or current_idx == total_count:
        progress_percent = (current_idx / total_count) * 100
        print(f">>> [CPU Transcoding Progress] {current_idx}/{total_count} ({progress_percent:.1f}%)")

    # Execute FFmpeg command to convert audio
    # -y: overwrite output files without asking
    # -ar: set audio sample rate
    # -ac: set number of audio channels (1 for mono)
    # -vn: disable video recording
    subprocess.run([
        "ffmpeg", "-y", "-i", str(file_path),
        "-ar", str(model_sr), "-ac", "1", "-vn", str(temp_wav_path)
    ], capture_output=True)

    return temp_wav_path


def extract_embeddings_vinet_fast(file_paths, output_path, config_path, granularity="track"):
    """
    Orchestrates the parallel transcoding of audio files followed by
    GPU-based embedding extraction using the VINet model.
    """
    model_sr = 44100

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        total_files = len(file_paths)
        print(f"Starting parallel transcoding of {total_files} files to temporary directory...")

        # Prepare tasks for parallel execution, including index for progress tracking
        tasks = [(fp, tmp_dir, model_sr, i + 1, total_files) for i, fp in enumerate(file_paths)]

        # Use ThreadPoolExecutor for CPU-bound I/O tasks (FFmpeg calls)
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Execute all tasks and wait for completion
            list(executor.map(convert_one_file, tasks))


        print("--- Transcoding complete! Loading VINet model for GPU extraction ---")

        # Create a copy of the current environment variables and update PYTHONPATH.
        # Core Fix 1: Add the absolute path of VINet to the environment variables
        # to prevent import errors when inference.py executes.
        env = os.environ.copy()
        env["PYTHONPATH"] = "/app/models/Discogs-VINet:/app:" + env.get("PYTHONPATH", "")

        cmd = [
            "python",
            # Core Fix 2: Use the absolute path to precisely locate inference.py.
            "/app/models/Discogs-VINet/inference.py",
            str(tmp_dir_path),
            config_path,
            str(output_path),
            f"--granularity={granularity}"
        ]


        # Execute the inference script with the updated environment and working directory.
        result = subprocess.run(cmd, cwd="/app/models/Discogs-VINet", env=env)


        if result.returncode != 0:
            print("VINet batch inference failed!")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings using Discogs-VINet")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to text file containing list of input audio files")
    parser.add_argument("--output-path", type=str, required=True, help="Directory path to save output embeddings")
    parser.add_argument("--config", type=str, default="/app/models/Discogs-VINet/logs/checkpoints/Discogs-VINet-MIREX-full_set/config.yaml",
                        help="Path to the model configuration file")
    args = parser.parse_args()

    # Read input file paths, handling UTF-8 BOM and filtering empty lines
    with open(args.input, "r", encoding="utf-8-sig") as f:
        path_list = [Path(line.strip()) for line in f if line.strip()]

    # Filter list to include only existing files
    valid_files = [p for p in path_list if p.is_file()]

    # Ensure output directory exists
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Execute the extraction pipeline
    extract_embeddings_vinet_fast(valid_files, output_path, args.config)

    print("Discogs-VINet extraction task completed successfully!")


if __name__ == "__main__":
    # Configure basic logging settings
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
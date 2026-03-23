#!/usr/bin/env python3
"""Demucs source separation helper.

Install Demucs first:
    pip install -U demucs

Examples:
    # 4-stem separation (vocals, drums, bass, other)
    python utils/demucs_separate.py my_song.wav --model htdemucs --output-dir separated

    # Vocals-only separation
    python utils/demucs_separate.py my_song.wav --vocals-only --model htdemucs

    # Force CPU
    python utils/demucs_separate.py my_song.wav --device cpu

    # Python API usage
    from utils.demucs_separate import separate_track
    stems = separate_track("my_song.wav", output_dir="separated")
    print(stems["vocals"])  # path to vocals stem file
"""

import argparse
from pathlib import Path
from typing import Dict

import demucs.separate


def separate_track(
    input_file: str,
    model: str = "htdemucs",
    vocals_only: bool = False,
    mp3: bool = False,
    output_dir: str = "separated",
    device: str | None = None,
) -> Dict[str, str]:
    """Run Demucs on one track and return separated stem file paths.

    Returns a mapping like:
      {"vocals": "...", "drums": "...", "bass": "...", "other": "..."}
    For vocals-only mode, only the "vocals" key is returned.
    """
    input_path = Path(input_file).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    args: list[str] = ["-n", model, "-o", str(output_path)]

    if vocals_only:
        args.extend(["--two-stems", "vocals"])

    if mp3:
        args.append("--mp3")

    if device:
        args.extend(["-d", device])

    args.append(str(input_path))
    demucs.separate.main(args)

    track_name = input_path.stem
    extension = "mp3" if mp3 else "wav"
    base = output_path / model / track_name

    if vocals_only:
        return {"vocals": str(base / f"vocals.{extension}")}

    return {
        "vocals": str(base / f"vocals.{extension}"),
        "drums": str(base / f"drums.{extension}"),
        "bass": str(base / f"bass.{extension}"),
        "other": str(base / f"other.{extension}"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run music source separation with Demucs.")
    parser.add_argument("input_file", help="Path to input audio file (e.g., my_song.wav)")
    parser.add_argument(
        "--model",
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"],
        help="Pretrained model to use (default: htdemucs)",
    )
    parser.add_argument(
        "--vocals-only",
        action="store_true",
        help="Extract only vocals with --two-stems vocals",
    )
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Save outputs as mp3",
    )
    parser.add_argument(
        "--output-dir",
        default="separated",
        help="Directory where Demucs writes separated stems (default: separated)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device override (e.g., cpu if GPU memory is limited)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stems = separate_track(
        input_file=args.input_file,
        model=args.model,
        vocals_only=args.vocals_only,
        mp3=args.mp3,
        output_dir=args.output_dir,
        device=args.device,
    )

    print("Created stems:")
    for stem_name, stem_path in stems.items():
        print(f"- {stem_name}: {stem_path}")


if __name__ == "__main__":
    main()

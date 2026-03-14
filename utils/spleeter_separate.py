#!/usr/bin/env python3
"""Spleeter source separation helper.

Install Spleeter first:
    pip install -U spleeter

Examples:
    # 4-stem separation (vocals, drums, bass, other)
    python utils/spleeter_separate.py my_song.wav --output-dir separated

    # Vocals-only separation (using Spleeter 2-stem model)
    python utils/spleeter_separate.py my_song.wav --vocals-only --output-dir separated

    # Python API usage
    from utils.spleeter_separate import separate_track
    stems = separate_track("my_song.wav", output_dir="separated")
    print(stems["vocals"])  # path to vocals stem file
"""

import argparse
from pathlib import Path
from typing import Dict

from spleeter.separator import Separator


def separate_track(
    input_file: str,
    vocals_only: bool = False,
    output_dir: str = "separated",
) -> Dict[str, str]:
    """Run Spleeter on one track and return separated stem file paths.

    Returns a mapping like:
      {"vocals": "...", "drums": "...", "bass": "...", "other": "..."}
    For vocals-only mode, only the "vocals" key is returned.
    """
    input_path = Path(input_file).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    model_name = "spleeter:2stems" if vocals_only else "spleeter:4stems"
    separator = Separator(model_name)
    separator.separate_to_file(str(input_path), str(output_path))

    track_output_dir = output_path / input_path.stem

    if vocals_only:
        return {"vocals": str(track_output_dir / "vocals.wav")}

    return {
        "vocals": str(track_output_dir / "vocals.wav"),
        "drums": str(track_output_dir / "drums.wav"),
        "bass": str(track_output_dir / "bass.wav"),
        "other": str(track_output_dir / "other.wav"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run music source separation with Spleeter.")
    parser.add_argument("input_file", help="Path to input audio file (e.g., my_song.wav)")
    parser.add_argument(
        "--vocals-only",
        action="store_true",
        help="Extract only vocals with the 2-stem model",
    )
    parser.add_argument(
        "--output-dir",
        default="separated",
        help="Directory where Spleeter writes separated stems (default: separated)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stems = separate_track(
        input_file=args.input_file,
        vocals_only=args.vocals_only,
        output_dir=args.output_dir,
    )

    print("Created stems:")
    for stem_name, stem_path in stems.items():
        print(f"- {stem_name}: {stem_path}")


if __name__ == "__main__":
    main()

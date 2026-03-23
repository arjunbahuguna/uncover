import argparse
import subprocess
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
from pedalboard import Pedalboard, PitchShift


def _write_audio(path, y, sr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y.T if y.ndim > 1 else y, sr)


def _decode_input_to_temp_wav(input_path: str) -> tuple[tempfile.TemporaryDirectory, str]:
    """Decode arbitrary input media to a temporary WAV via ffmpeg."""
    temp_dir = tempfile.TemporaryDirectory(prefix="pitch_shift_")
    temp_wav = str(Path(temp_dir.name) / "decoded_input.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        temp_wav,
    ]
    process = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        temp_dir.cleanup()
        raise RuntimeError(
            "ffmpeg failed to decode input media to WAV. "
            f"input={input_path}\n{process.stderr}"
        )
    return temp_dir, temp_wav


class PitchShiftTool:
    def __init__(self):
        pass

    def process(self, input_path, output_path, n_steps=0.0):
        # input_path: Source audio file
        # output_path: Destination audio file
        # n_steps: Number of semitones to shift (e.g., 2.0 or -2.0)
        temp_dir, decoded_input_wav = _decode_input_to_temp_wav(str(input_path))

        try:
            if n_steps == 0.0:
                y, sr = librosa.load(decoded_input_wav, sr=None, mono=False)
                _write_audio(output_path, y, sr)
                print(f"Skipped (n_steps=0.0) -> {output_path}")
                return

            y, sr = librosa.load(decoded_input_wav, sr=None, mono=False)

            # pedalboard (JUCE-based, cross-platform, no external binaries)
            print(f"Pitch Shifting: {n_steps} semitones")
            board = Pedalboard([PitchShift(semitones=n_steps)])
            y_shifted = board(y, sr)

            _write_audio(output_path, y_shifted, sr)
            print(f"Completed -> {output_path}")
        finally:
            temp_dir.cleanup()


def build_parser():
    parser = argparse.ArgumentParser(description="Apply pitch shift to an audio file.")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", "--output-path", dest="output", required=True, help="Path to output audio file")
    parser.add_argument("--n-steps", type=float, default=0.0, help="Pitch shift in semitones")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    ps_tool = PitchShiftTool()
    ps_tool.process(args.input, args.output, n_steps=args.n_steps)
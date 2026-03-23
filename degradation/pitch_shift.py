import argparse

import librosa
import soundfile as sf
from pedalboard import Pedalboard, PitchShift


def _write_audio(path, y, sr):
    sf.write(path, y.T if y.ndim > 1 else y, sr)


class PitchShiftTool:
    def __init__(self):
        pass

    def process(self, input_path, output_path, n_steps=0.0):
        # input_path: Source audio file
        # output_path: Destination audio file
        # n_steps: Number of semitones to shift (e.g., 2.0 or -2.0)

        if n_steps == 0.0:
            y, sr = librosa.load(input_path, sr=None, mono=False)
            _write_audio(output_path, y, sr)
            print(f"Skipped (n_steps=0.0) -> {output_path}")
            return

        y, sr = librosa.load(input_path, sr=None, mono=False)

        # pedalboard (JUCE-based, cross-platform, no external binaries)
        print(f"Pitch Shifting: {n_steps} semitones")
        board = Pedalboard([PitchShift(semitones=n_steps)])
        y_shifted = board(y, sr)

        _write_audio(output_path, y_shifted, sr)
        print(f"Completed -> {output_path}")


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
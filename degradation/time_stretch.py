import argparse

import numpy as np
import librosa
import soundfile as sf


def _write_audio(path, y, sr):
    sf.write(path, y.T if y.ndim > 1 else y, sr)


class TimeStretchTool:
    def __init__(self):
        pass

    def process(self, input_path, output_path, stretch_rate=1.0):
        # input_path: Source audio file
        # output_path: Destination audio file
        # stretch_rate: Speed factor (>1 faster, <1 slower)

        if stretch_rate == 1.0:
            y, sr = librosa.load(input_path, sr=None, mono=False)
            _write_audio(output_path, y, sr)
            print(f"Skipped (rate=1.0) -> {output_path}")
            return

        y, sr = librosa.load(input_path, sr=None, mono=False)

        print(f"Time Stretching: {stretch_rate}x")
        if y.ndim > 1:
            channels = [librosa.effects.time_stretch(y[c], rate=stretch_rate, n_fft=4096) for c in range(y.shape[0])]
            y_stretched = np.stack(channels, axis=0)
        else:
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate, n_fft=4096)

        _write_audio(output_path, y_stretched, sr)
        print(f"Completed -> {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Apply time stretch to an audio file.")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", "--output-path", dest="output", required=True, help="Path to output audio file")
    parser.add_argument("--stretch-rate", type=float, default=1.0, help="Stretch factor (>1 faster, <1 slower)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    ts_tool = TimeStretchTool()
    ts_tool.process(args.input, args.output, stretch_rate=args.stretch_rate)
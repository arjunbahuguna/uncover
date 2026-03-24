import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

"""
This file provides a batch time-stretch utility with selectable backend (librosa or audiomentations). 

Example usage: python degradation/time_stretch.py --input_path /path/to/input/folder --output_path /path/to/output/folder --backend audiomentations --stretch-rates 0.5,0.8,1.0,1.2,1.5,2.0

"""


def _write_audio(path, y, sr):
    """Write mono or multi-channel audio to disk.

    librosa returns arrays as (samples,) for mono or (channels, samples) for
    multi-channel when mono=False. soundfile expects (samples,) or
    (samples, channels), so we transpose multi-channel arrays before writing.
    """
    sf.write(path, y.T if y.ndim > 1 else y, sr)


def _decode_input_to_temp_wav(input_path: str) -> tuple[tempfile.TemporaryDirectory, str]:
    """Decode arbitrary input media to a temporary WAV via ffmpeg."""
    # Normalize path separators to forward slashes for cross-platform compatibility
    clean_path = input_path.replace('\\', '/')

    # Ensure the path is absolute by prepending a root slash if missing
    # This assumes the data resides under the root directory; alternatively, use Path(clean_path).absolute()
    if not clean_path.startswith('/'):
        clean_path = '/' + clean_path.lstrip('/')

    # Create a temporary directory with a specific prefix for time-stretch operations
    temp_dir = tempfile.TemporaryDirectory(prefix="time_stretch_")
    temp_wav = str(Path(temp_dir.name) / "decoded_input.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        clean_path,
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


def _parse_rates(rates_arg: str):
    """Parse a comma-separated list of positive stretch rates."""
    rates = []
    for chunk in rates_arg.split(","):
        value = float(chunk.strip())
        if value <= 0:
            raise ValueError(f"Invalid stretch rate {value}. Rates must be > 0.")
        rates.append(value)
    return rates


def _format_rate_for_filename(rate: float):
    """Create a filesystem-safe rate suffix (e.g. 1.2 -> 1p2)."""
    return f"{rate:g}".replace(".", "p")


def _iter_audio_files(input_root: Path):
    """Yield supported audio files recursively from input_root."""
    supported_exts = {".wav", ".mp3"}
    for path in input_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_exts:
            yield path


def _output_ext_for_input(input_ext: str):
    """Return output extension.

    MP3 writing is not reliably available through soundfile in all setups, so
    we standardize outputs to WAV.
    """
    # soundfile reliably writes WAV. MP3 inputs are converted to WAV outputs.
    return ".wav"


def _build_failure_logger(log_path: Path):
    """Create a dedicated logger that writes failed augmentations to a file."""
    logger = logging.getLogger("time_stretch_failures")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if process_folder is invoked multiple times.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


class TimeStretchTool:
    """Batch time-stretch utility with selectable backend.

    Supported backends:
    - librosa: uses librosa.effects.time_stretch
    - audiomentations: uses audiomentations.TimeStretch with fixed min/max rate
    """

    def __init__(self, backend="librosa"):
        backend = backend.lower()
        if backend not in {"librosa", "audiomentations"}:
            raise ValueError("backend must be one of: librosa, audiomentations")
        self.backend = backend
        self._audiomentations_time_stretch = None

        if self.backend == "audiomentations":
            try:
                from audiomentations import TimeStretch as AudioMentationsTimeStretch
            except ImportError as exc:
                raise ImportError(
                    "audiomentations backend selected but package is not installed. "
                    "Install it with: pip install audiomentations"
                ) from exc
            self._audiomentations_time_stretch = AudioMentationsTimeStretch

    def _stretch_librosa(self, y, rate):
        if rate == 1.0:
            return y

        if y.ndim > 1:
            channels = [
                librosa.effects.time_stretch(y[c], rate=rate)
                for c in range(y.shape[0])
            ]
            return np.stack(channels, axis=0)

        return librosa.effects.time_stretch(y, rate=rate)

    def _stretch_audiomentations(self, y, sr, rate):
        """Stretch with audiomentations in the shape format it expects.

        audiomentations expects:
        - mono: (num_samples,)
        - multi-channel: (num_samples, num_channels)
        """
        if rate == 1.0:
            return y

        transform = self._audiomentations_time_stretch(
            min_rate=rate,
            max_rate=rate,
            leave_length_unchanged=False,
            p=1.0,
        )

        samples = np.ascontiguousarray(y.T if y.ndim > 1 else y, dtype=np.float32)
        stretched = transform(samples=samples, sample_rate=sr)

        if y.ndim > 1:
            if stretched.ndim != 2:
                raise ValueError(
                    "Unexpected audiomentations output shape for multi-channel "
                    f"input: {stretched.shape}"
                )
            return np.ascontiguousarray(stretched.T, dtype=np.float32)

        return np.ascontiguousarray(stretched, dtype=np.float32)

    def process(self, input_path, output_path, stretch_rate=1.0):
        y, sr = librosa.load(input_path, sr=None, mono=False)
        y = y.astype(np.float32, copy=False)

        if self.backend == "librosa":
            y_stretched = self._stretch_librosa(y, stretch_rate)
        else:
            y_stretched = self._stretch_audiomentations(y, sr, stretch_rate)

        _write_audio(output_path, y_stretched, sr)

    def process_folder(self, input_path, output_path, stretch_rates, fail_log_path=None):
        """Process all supported files under input_path with all stretch rates.

        Failures are logged to a file and processing continues for remaining
        files/rates.
        """
        input_root = Path(input_path)
        output_root = Path(output_path)

        if not input_root.exists() or not input_root.is_dir():
            raise ValueError(f"Input path does not exist or is not a directory: {input_root}")

        output_root.mkdir(parents=True, exist_ok=True)
        log_path = Path(fail_log_path) if fail_log_path else (output_root / "time_stretch_failures.log")
        fail_logger = _build_failure_logger(log_path)

        processed = 0
        failed = 0
        for input_file in _iter_audio_files(input_root):
            rel = input_file.relative_to(input_root)
            out_dir = output_root / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            output_ext = _output_ext_for_input(input_file.suffix.lower())
            for rate in stretch_rates:
                rate_tag = _format_rate_for_filename(rate)
                output_name = f"{input_file.stem}_ts_{rate_tag}{output_ext}"
                output_file = out_dir / output_name

                try:
                    self.process(str(input_file), str(output_file), stretch_rate=rate)
                    processed += 1
                    print(f"[{self.backend}] {input_file} -> {output_file} (rate={rate})")
                except Exception as exc:
                    failed += 1
                    fail_logger.exception(
                        "FAILED input=%s output=%s rate=%s backend=%s error=%s",
                        input_file,
                        output_file,
                        rate,
                        self.backend,
                        exc,
                    )
                    print(
                        f"[FAILED] {input_file} -> {output_file} "
                        f"(rate={rate}) | {exc}"
                    )

        print(
            f"Done. Generated {processed} files in: {output_root}. "
            f"Failures: {failed}."
        )
        print(f"Failure log: {log_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Batch time-stretch audio files found under input_path and save "
            "augmented outputs under output_path."
        )
    )
    parser.add_argument(
        "--input_path",
        "--input-path-audio",
        dest="input_path",
        required=False,
        help="Input root folder containing .wav/.mp3 files",
    )
    parser.add_argument(
        "--output_path",
        "--output-path-audio",
        dest="output_path",
        required=False,
        help="Output root folder for augmented audio files",
    )
    parser.add_argument(
        "--backend",
        choices=["librosa", "audiomentations"],
        default="librosa",
        help="Time-stretch backend to use",
    )
    parser.add_argument(
        "--stretch-rates",
        type=str,
        default="0.5,0.8,1.0,1.2,1.5,2.0",
        help="Comma-separated fixed stretch rates, e.g. 0.5,0.8,1.2,1.5,2",
    )
    parser.add_argument(
        "--fail-log-path",
        type=str,
        default=None,
        help=(
            "Optional path for failures log file. Default: "
            "<output_path>/time_stretch_failures.log"
        ),
    )
    parser.add_argument(
        "--input",
        dest="single_input",
        type=str,
        default=None,
        help="Single-file mode: path to one input audio file.",
    )
    parser.add_argument(
        "--output",
        dest="single_output",
        type=str,
        default=None,
        help="Single-file mode: path to output audio file (recommended .wav).",
    )
    parser.add_argument(
        "--stretch-rate",
        dest="single_stretch_rate",
        type=float,
        default=1.0,
        help="Single-file mode: time-stretch rate (> 0).",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if (args.single_input is None) != (args.single_output is None):
        raise ValueError("Single-file mode requires both --input and --output")

    if args.single_input is not None:
        if args.single_stretch_rate <= 0:
            raise ValueError("--stretch-rate must be > 0")

        temp_dir, decoded_input_wav = _decode_input_to_temp_wav(args.single_input)
        try:
            ts_tool = TimeStretchTool(backend=args.backend)
            ts_tool.process(
                input_path=decoded_input_wav,
                output_path=args.single_output,
                stretch_rate=args.single_stretch_rate,
            )
        finally:
            temp_dir.cleanup()
        print(
            f"[{args.backend}] {args.single_input} -> {args.single_output} "
            f"(rate={args.single_stretch_rate})"
        )
    else:
        if not args.input_path or not args.output_path:
            raise ValueError(
                "Batch mode requires --input_path and --output_path. "
                "For one file, use --input and --output."
            )
        rates = _parse_rates(args.stretch_rates)
        ts_tool = TimeStretchTool(backend=args.backend)
        ts_tool.process_folder(
            input_path=args.input_path,
            output_path=args.output_path,
            stretch_rates=rates,
            fail_log_path=args.fail_log_path,
        )
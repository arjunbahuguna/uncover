import argparse
import random
import librosa
import soundfile as sf
from pathlib import Path
from pedalboard import Pedalboard, Reverb, Convolution


def apply_reverb_advanced(input_path, output_path, mode='algo', ir_path=None, wet_level=0.3):
    """
    Apply reverb to audio using algorithmic or convolution methods.

    Args:
        input_path (str): Path to input audio file.
        output_path (str): Path to save output audio.
        mode (str): 'algo' for algorithmic, 'ir' for convolution.
        ir_path (str, optional): Path to IR file or directory. Required for 'ir' mode.
        wet_level (float): Reverb intensity (0.0 - 1.0).

    Returns:
        str: Selected IR path or "algorithmic".
    """
    # Load audio
    y, sr = librosa.load(input_path, sr=None)

    # Initialize effect
    if mode == 'ir':
        if not ir_path:
            raise ValueError("ir_path required for IR mode")

        p = Path(ir_path)
        if p.is_dir():
            ir_files = list(p.rglob("*.wav"))
            if not ir_files:
                raise FileNotFoundError(f"No .wav IR files found in {ir_path}")
            selected_ir = str(random.choice(ir_files))
            print(f"[IR Mode] Selected: {selected_ir}")
        else:
            selected_ir = str(p)
            print(f"[IR Mode] Using: {selected_ir}")

        effect = Convolution(selected_ir, mix=wet_level)

    else:
        print(f"[Algo Mode] Wet: {wet_level}")
        effect = Reverb(
            room_size=0.75,
            damping=0.5,
            wet_level=wet_level,
            dry_level=1.0 - (wet_level / 2)
        )

    # Process audio
    board = Pedalboard([effect])
    effected_audio = board(y, sr)

    # Save output
    sf.write(output_path, effected_audio.T, sr)
    return selected_ir if mode == 'ir' else "algorithmic"


def build_parser():
    parser = argparse.ArgumentParser(description="Apply reverb to an audio file.")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", "--output-path", dest="output", required=True, help="Path to output audio file")
    parser.add_argument("--mode", choices=["algo", "ir"], default="algo", help="Reverb mode")
    parser.add_argument("--ir-path", help="IR file or directory path (required when --mode ir)")
    parser.add_argument("--wet-level", type=float, default=0.3, help="Wet mix amount between 0.0 and 1.0")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.mode == "ir" and not args.ir_path:
        raise ValueError("--ir-path is required when --mode ir")

    apply_reverb_advanced(
        args.input,
        args.output,
        mode=args.mode,
        ir_path=args.ir_path,
        wet_level=args.wet_level,
    )
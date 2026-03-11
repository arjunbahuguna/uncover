import os
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


if __name__ == "__main__":
    # Example usage (uncomment to run):
    apply_reverb_advanced(r"Z:\CloudMusic\Enchanted (Taylor's Version).mp3", "out_algo.wav", mode='algo', wet_level=0.8)
    apply_reverb_advanced(r"Z:\CloudMusic\Enchanted (Taylor's Version).mp3", "out_ir.wav", mode='ir', ir_path="IRlib", wet_level=0.8)
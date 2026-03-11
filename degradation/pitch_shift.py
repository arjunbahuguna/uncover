import os
import subprocess
import librosa
import soundfile as sf


class PitchShiftTool:
    def __init__(self, rb_path="rubberband.exe"):
        # Path to rubberband executable
        self.rb_path = rb_path
        self.has_rb = os.path.exists(rb_path)
        if not self.has_rb:
            print(f"Warning: Executable not found at {rb_path}. High-quality mode disabled.")

    def process(self, input_path, output_path, n_steps=0.0, quality='high'):
        # input_path: Source audio file
        # output_path: Destination audio file
        # n_steps: Number of semitones to shift (e.g., 2.0 or -2.0)
        # quality: 'high' (Rubber Band) or 'low' (Librosa)

        if n_steps == 0.0:
            # Copy file without processing
            y, sr = librosa.load(input_path, sr=None)
            sf.write(output_path, y, sr)
            print(f"Skipped (n_steps=0.0) -> {output_path}")
            return

        if quality == 'high' and self.has_rb:
            # High-quality mode (Rubber Band)
            # -p specifies the pitch shift in semitones
            cmd = [self.rb_path, "-p", str(n_steps), input_path, output_path]

            print(f"[High Quality] Pitch Shifting: {n_steps} semitones")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        else:
            # Low-quality mode (Librosa)
            print(f"[Low Quality] Pitch Shifting: {n_steps} semitones")
            y, sr = librosa.load(input_path, sr=None)
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            sf.write(output_path, y_shifted, sr)

        print(f"Completed -> {output_path}")


if __name__ == "__main__":
    ps_tool = PitchShiftTool(rb_path="rubberband.exe")

    # High-quality: Shift up by 2 semitones
    ps_tool.process(r"Z:\CloudMusic\Enchanted (Taylor's Version).mp3", "pitch_up_high.wav", n_steps=2.0, quality='high')

    # Low-quality: Shift down by 2 semitones
    ps_tool.process(r"Z:\CloudMusic\Enchanted (Taylor's Version).mp3", "pitch_up_low.wav", n_steps=2.0, quality='low')
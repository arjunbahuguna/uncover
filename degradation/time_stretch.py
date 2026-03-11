import os
import subprocess
import librosa
import soundfile as sf


class TimeStretchTool:
    def __init__(self, rb_path="rubberband.exe"):
        # Path to rubberband executable
        self.rb_path = rb_path
        self.has_rb = os.path.exists(rb_path)
        if not self.has_rb:
            print(f"Warning: Executable not found at {rb_path}. High-quality mode disabled.")

    def process(self, input_path, output_path, stretch_rate=1.0, quality='high'):
        # input_path: Source audio file
        # output_path: Destination audio file
        # stretch_rate: Speed factor (>1 faster, <1 slower)
        # quality: 'high' (Rubber Band) or 'low' (Librosa)

        if stretch_rate == 1.0:
            # Copy file without processing
            y, sr = librosa.load(input_path, sr=None)
            sf.write(output_path, y, sr)
            print(f"Skipped (rate=1.0) -> {output_path}")
            return

        if quality == 'high' and self.has_rb:
            # High-quality mode (Rubber Band)
            # time_ratio = 1.0 / stretch_rate
            time_ratio = 1.0 / stretch_rate
            cmd = [self.rb_path, "-t", str(time_ratio), input_path, output_path]

            print(f"[High Quality] Stretching: {stretch_rate}x")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        else:
            # Low-quality mode (Librosa Phase Vocoder)
            print(f"[Low Quality] Stretching: {stretch_rate}x")
            y, sr = librosa.load(input_path, sr=None)
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
            sf.write(output_path, y_stretched, sr)

        print(f"Completed -> {output_path}")


if __name__ == "__main__":
    ts_tool = TimeStretchTool(rb_path="rubberband.exe")

    # High-quality 1.2x speedup
    ts_tool.process(r"Z:\CloudMusic\Enchanted (Taylor's Version).mp3", "fast_high.wav", stretch_rate=1.8, quality='high')

    # Low-quality 0.8x slowdown
    ts_tool.process(r"Z:\CloudMusic\Enchanted (Taylor's Version).mp3", "fast_low.wav", stretch_rate=1.8, quality='low')
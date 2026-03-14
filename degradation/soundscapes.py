import scaper

# Soundbank folders
fg_path = "soundbank/foreground"
bg_path = "soundbank/background"

# Output folder
outfolder = "generated_soundscapes"

# Create scaper object
sc = scaper.Scaper(
    duration=20.0,        # length of soundscape
    fg_path=fg_path,
    bg_path=bg_path
)

# Add background music
sc.add_background(
    label=("choose", []), 
    source_file=("choose", []),
    source_time=("const", 0)
)

# Add music event
sc.add_event(
    label=("const", "music"),
    source_file=("choose", []),
    source_time=("uniform", 0, 10),
    event_time=("uniform", 0, 10),
    event_duration=("uniform", 10, 20),
    # SNR is in dB and applies to foreground events relative to background.
    # Negative values make music quieter than speech background.
    snr=("uniform", -12, -11),
    pitch_shift=None,
    time_stretch=None
)

# Generate soundscape
audiofile = outfolder + "/soundscape.wav"
jamsfile = outfolder + "/soundscape.jams"

sc.generate(audiofile, jamsfile)
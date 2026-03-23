import sys
import logging
import argparse
import os
import torch
import importlib
import tempfile
import subprocess
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Import internal CLEWS utility classes
# Assumes the current working directory is /app
sys.path.append("/app")
from utils import pytorch_utils, audio_utils

logger = logging.getLogger(__name__)


def load_clews_model(checkpoint_path, device="cuda"):
    """
    Loads the model into GPU memory in a single step.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        device (str): Device to load the model onto (default: "cuda").
    
    Returns:
        tuple: The loaded model and its configuration object.
    """
    logger.info(f"Loading model to {device}...")
    path_checkpoint = os.path.dirname(checkpoint_path)
    conf = OmegaConf.load(os.path.join(path_checkpoint, "configuration.yaml"))

    # 1. Initialize model architecture
    module = importlib.import_module("models." + conf.model.name)
    model = module.Model(conf.model, sr=conf.data.samplerate)

    # 2. Load weights file
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Core Logic: Identify and extract the actual model state dictionary
    if 'model' in ckpt:
        # Case: Weights are nested under the 'model' key
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        # Case: Standard PyTorch Lightning format
        state_dict = ckpt['state_dict']
    else:
        # Case: The checkpoint contains only the raw state dictionary
        state_dict = ckpt

    # 3. Load model weights (handling potential prefix issues)
    # Some saved models include a 'model.' prefix which needs to be removed
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[6:] if k.startswith('model.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()
    return model, conf


def process_audio_worker(file_path, model_sr, audio_queue):
    """
    CPU worker thread responsible for parallel transcoding and audio loading.
    
    Args:
        file_path (Path): Path to the input audio file.
        model_sr (int): Target sample rate for the model.
        audio_queue (queue.Queue): Queue to store processed tensors.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_wav_path = Path(tmp_dir) / "temp.wav"
        
        # Convert audio using ffmpeg: mono channel, no video, target sample rate
        subprocess.run([
            "ffmpeg", "-y", "-i", str(file_path),
            "-ar", str(model_sr), "-ac", "1", "-vn", str(temp_wav_path)
        ], capture_output=True)

        if temp_wav_path.exists():
            x = audio_utils.load_audio(str(temp_wav_path), sample_rate=model_sr, n_channels=1)
            # Push the processed tensor and original file path to the queue
            audio_queue.put((x, file_path))


def extract_embeddings_clews_fast(file_paths, output_path, checkpoint_path):
    """
    Main pipeline for extracting embeddings using producer-consumer pattern.
    Uses multi-threading for CPU-bound audio processing and GPU for inference.
    
    Args:
        file_paths (list): List of input file paths.
        output_path (Path): Directory to save output embeddings.
        checkpoint_path (str): Path to the model checkpoint.
    """
    device = "cuda"
    model, conf = load_clews_model(checkpoint_path, device)
    # Buffer queue to prevent memory overflow
    audio_queue = queue.Queue(maxsize=20)

    # 1. Start CPU thread pool for parallel transcoding
    # Uses 8 workers to maximize utilization on CPUs like Ryzen 5700X3D
    def producer():
        with ThreadPoolExecutor(max_workers=8) as executor:
            for fp in file_paths:
                executor.submit(process_audio_worker, fp, model.sr, audio_queue)
        # Send termination signal
        audio_queue.put((None, None))

    threading.Thread(target=producer, daemon=True).start()

    # 2. GPU inference main loop
    pbar = tqdm(total=len(file_paths), desc="Processing embeddings")
    with torch.inference_mode():
        while True:
            x, file_path = audio_queue.get()
            
            # Check for termination signal
            if x is None:
                break

            output_file = output_path / f"{file_path.stem}.pt"
            x = x.to(device)
            
            # Run inference
            z = model(x)
            
            # Save result (squeeze batch dimension, move to CPU)
            torch.save(z.squeeze(0).cpu(), output_file)
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description="Extract audio embeddings using CLEWS model.")
    parser.add_argument("--input", type=str, required=True, help="Path to text file containing list of input audio files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--output-path", type=str, required=True, help="Directory path to save output embedding files.")
    args = parser.parse_args()

    # Read file list from input text file
    with open(args.input, "r", encoding="utf-8-sig") as f:
        path_list = [Path(line.strip()) for line in f if line.strip()]

    # Filter valid existing files
    valid_files = [p for p in path_list if p.is_file()]
    logger.info(f"Number of valid files found: {len(valid_files)}")

    extract_embeddings_clews_fast(valid_files, Path(args.output_path), args.checkpoint)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
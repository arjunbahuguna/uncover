# UnCover
Unified evaluation pipeline for version identification models

## CLEWS Inference

The `clews` service runs the code from `models/clews` inside Docker.

Build the image:

```bash
make build-clews
```

Open a shell in the container:

```bash
make bash-clews
```

Run inference (inside the container) with a checkpoint and a single input file:

```bash
python inference.py --checkpoint=checkpoints/clews/dvi-clews/checkpoint_best.ckpt --fn_in=inputs/my_video.wav --fn_out=output/filename.pt --device=cpu
```

Notes:

- Use `.wav` input files for the most reliable decoding in this setup.
- Paths are relative to `/app` inside the container (`models/clews` on the host).
- The output embedding is saved as a `.pt` file at the path given by `--fn_out`.

## Discogs-VINet Inference

The `discogs-vinet` service runs the code from `models/Discogs-VINet` inside Docker.

Build the image:

```bash
make build-discogs-vinet
```

Prepare an input directory under `models/Discogs-VINet` and place one or more `.wav` files there. The inference script scans directories recursively and only looks for `.wav` files.

Run inference with the provided MIREX full-set checkpoint:

```bash
docker compose run --rm discogs-vinet bash -lc '\
source /opt/conda/etc/profile.d/conda.sh && \
conda activate discogs-vinet && \
python inference.py \
	inputs \
	logs/checkpoints/Discogs-VINet-MIREX-full_set/config.yaml \
	outputs \
'
```

Notes:

- `inputs` is a directory, not a single file path. For a single track, place that track alone in the directory.
- `outputs` will contain `.npy` embedding files with the same relative structure as the input tree.
- If your source audio is not `.wav`, convert it first.

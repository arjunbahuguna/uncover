# UnCover
Unified evaluation pipeline for version identification models

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

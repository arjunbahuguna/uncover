# UnCover
Unified evaluation pipeline for version identification models

## Embedding Extraction

Extraction is now done from inside the corresponding model container by running `extractor.py` with the matching model name.

The input file must be a txt file with one absolute audio path per line (for example under `/data/discogs/...`).

### CLEWS extraction

Build and enter the CLEWS container:

```bash
make build-clews
make bash-clews
```

Run extraction inside the container:

```bash
python extractor/extractor.py --input extractor/path_test.txt --model clews --output-path output_embeddings
```

### Discogs-VINet extraction

Build and enter the Discogs-VINet container:

```bash
make build-discogs-vinet
make bash-discogs-vinet
```

Run extraction inside the container:

```bash
python extractor/extractor.py --input extractor/path_test.txt --model discogs-vinet --output-path output_embeddings
```

Notes:


- `--input` points to a plain text file where each line is one audio file path.
- Use `--model clews` only in the CLEWS container and `--model discogs-vinet` only in the Discogs-VINet container.
- CLEWS writes `.pt` embeddings. Discogs-VINet writes `.npy` embeddings.
- `--output-path` is created automatically if it does not exist.


## Audio Degradations (CLI)

Run from the repo root:

```bash
python degradation/pitch_shift.py --input in.wav --output out_pitch.wav --n-steps 2
python degradation/time_stretch.py --input in.wav --output out_stretch.wav --stretch-rate 1.2
python degradation/reverb.py --input in.wav --output out_reverb.wav --mode algo --wet-level 0.4
python degradation/reverb.py --input in.wav --output out_reverb_ir.wav --mode ir --ir-path path/to/ir_or_ir_folder --wet-level 0.4
```

Notes:

- `--input` and `--output` are required for all degradation scripts.


## Retrieval Evaluation (mAP, MR1, NAR, R@K)

Run from the repo root:

```bash
python retrieval/eval_retrieval.py \
	--first-list extractor/first_embeddings.txt \
	--second-list extractor/second_embeddings.txt \
	--metadata-json /data/discogs_test_subset.json \
	--dim 1024 \
	--k 1 10 100 \
	--verbose
```

Notes:

- `--first-list` is the database embeddings list.
- `--second-list` is the query embeddings list.
- `--metadata-json` must contain `version_id -> [{youtube_id, ...}]` mappings.
- The script computes a full pairwise L2 distance matrix, then evaluates using `eval/eval.py` metrics: mAP, MR1, NAR, and R@K.

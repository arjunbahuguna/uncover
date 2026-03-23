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

Build and enter the degradation container:

```bash
make build-degradation
make bash-degradation
```

Run from the repo root on the host or inside the container:

```bash
python degradation/pitch_shift.py --input in.wav --output out_pitch.wav --n-steps 2
python degradation/time_stretch.py --input-path in_dir --output-path out_dir --stretch-rates 1.2
python degradation/reverb.py --input in.wav --output out_reverb.wav --mode algo --wet-level 0.4
python degradation/reverb.py --input in.wav --output out_reverb_ir.wav --mode ir --ir-path path/to/ir_or_ir_folder --wet-level 0.4
```

Notes:

- `pitch_shift.py` and `reverb.py` require `--input` and `--output`.
- `time_stretch.py` works on directories via `--input-path` and `--output-path`.
- Docker usage follows the same pattern as the other services, for example: `docker compose run --rm degradation python degradation/pitch_shift.py --input /data/discogs/example.wav --output /data/discogs/example_pitch.wav --n-steps 2`.


## Retrieval Evaluation (mAP, MR1, NAR, R@K)

Run from the repo root:

```bash
python retrieval/eval_retrieval.py \
	--first-list extractor/first_embeddings.txt \
	--second-list extractor/second_embeddings.txt \
	--embedding-model clews \
	--metadata-json /data/discogs_test_subset.json \
	--k 1 10 100 \
	--verbose
```

Notes:

- `--first-list` is the database embeddings list.
- `--second-list` is the query embeddings list.
- `--embedding-model` selects the retrieval configuration automatically:
  - `clews`: dimension `1024`, metric `l2`
  - `discogs-vinet`: dimension `512`, metric `ip`, L2 normalization enabled
- `--metadata-json` must contain `version_id -> [{youtube_id, ...}]` mappings.
- The script computes a pairwise score matrix using the model-specific retrieval config, then evaluates using `eval/eval.py` metrics: mAP, MR1, NAR, and R@K.


## End-to-End Orchestrator Pipeline

Run from the repo root:

```bash
python pipeline_orchestrator.py \
	--input-json test-json.json \
	--embedding-model discogs-vinet \
	--enable-time-stretch-augmentation \
	--time-stretch-rate 1.2 \
	--docker-build-first \
	--output-dir extractor/.pipeline_runtime/discogs_vinet_run
```

What it does:

- Reads a JSON in the format `{work_id: [recording entries...]}`.
- Randomly chooses one recording per work for index and one for query.
- Optionally creates augmented query files (pitch shift or time stretch) and uses those augmented files as the only queries for evaluation.
- Runs embedding extraction with `extractor/extractor.py`.
- Builds index/query lists and evaluates retrieval with mAP, MR1, NAR, and R@K.
- Prints metrics and saves a full JSON report.

Notes:

- Recording entries must be strings (audio paths) or dicts with `path`/`audio_path`/`recording_path`/`file_path`/`filepath`.
- Relative paths in the JSON are resolved from the repository root.
- The script always runs extraction with `docker compose run --rm <service> python extractor/extractor.py ...`.
- The script runs retrieval evaluation in Docker too: `docker compose run --rm retrieval python retrieval/eval_retrieval.py ...`.
- `--enable-pitch-shift-augmentation` applies `degradation/pitch_shift.py` to each selected query and evaluates retrieval using only the augmented queries.
- `--pitch-shift-n-steps` controls semitone shift for pitch-shifted query files.
- `--enable-time-stretch-augmentation` applies `degradation/time_stretch.py` to each selected query and evaluates retrieval using only the augmented queries.
- `--time-stretch-rate` controls time-stretch factor for time-stretched query files.
- Use only one augmentation mode per run (`--enable-pitch-shift-augmentation` or `--enable-time-stretch-augmentation`).
- Use `--docker-build-first` if you want to rebuild the model image before extraction.
- `--output-dir` must be inside `extractor/` so container and host share generated files.
- The script automatically writes embeddings to `<output-dir>/embeddings` and report JSON to `<output-dir>/report.json`.
- If `<output-dir>/embeddings` already contains extracted files, only missing embeddings are extracted.
- Augmented query files are written under `extractor/.pipeline_runtime/<run>/augmented_queries` and are automatically converted to `/app/extractor/...` paths for the extractor container.

`retrieval/eval_retrieval.py` is modularized and supports:

- `--embedding-model` to select dimension/metric/normalization from the model name.
- `--metadata-json` for the original metadata format.
- `--labels-json` for explicit `embedding_stem -> {work_id, song_id}` labels.
- `--output-json` to save metrics/details for orchestration.

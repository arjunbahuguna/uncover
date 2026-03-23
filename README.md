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
	--docker-build-first \
	--output-dir extractor/.pipeline_runtime/discogs_vinet_run
```

What it does:

- Reads a JSON in the format `{work_id: [recording entries...]}`.
- Randomly chooses one recording per work for index and one for query.
- Runs embedding extraction with `extractor/extractor.py`.
- Builds index/query lists and evaluates retrieval with mAP, MR1, NAR, and R@K.
- Prints metrics and saves a full JSON report.

Notes:

- Recording entries must be strings (audio paths) or dicts with `path`/`audio_path`/`recording_path`/`file_path`/`filepath`.
- Relative paths in the JSON are resolved from the repository root.
- The script always runs extraction with `docker compose run --rm <service> python extractor/extractor.py ...`.
- The script runs retrieval evaluation in Docker too: `docker compose run --rm retrieval python retrieval/eval_retrieval.py ...`.
- Use `--docker-build-first` if you want to rebuild the model image before extraction.
- `--output-dir` must be inside `extractor/` so container and host share generated files.
- The script automatically writes embeddings to `<output-dir>/embeddings` and report JSON to `<output-dir>/report.json`.
- If `<output-dir>/embeddings` already contains extracted files, only missing embeddings are extracted.
- In Docker mode, host audio paths are mapped using the repository docker-compose mount: `/Volumes/T7 Shield/discogs` -> `/data/discogs`.

`retrieval/eval_retrieval.py` is modularized and supports:

- `--embedding-model` to select dimension/metric/normalization from the model name.
- `--metadata-json` for the original metadata format.
- `--labels-json` for explicit `embedding_stem -> {work_id, song_id}` labels.
- `--output-json` to save metrics/details for orchestration.

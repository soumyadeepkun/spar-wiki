# Spar-Wiki

Efficient sparse indexing pipeline and retrieval API over Wikipedia abstracts.

## Overview

This repository provides:

- A SQLite seeding script for document metadata.
- A SPLADE-based sparse indexing pipeline.
- A FastAPI search API for single and batch retrieval.
- A minimal [`dspy.ColBERTv2`](https://dspy.ai/#__tabbed_2_2:~:text=Math-,RAG,-Classification) example client (plug-in splade index replacement for the colbert index).

## Prerequisites

- `uv` installed.
- Python `>=3.12`.
- A Hugging Face token with access to [`naver/splade-v3`](https://huggingface.co/naver/splade-v3).
- Input TSV/CSV data containing `id`, `title`, and `text` columns.

## Local Setup (uv)

```bash
# from repo root
uv python install 3.12
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync --no-dev
```

## Environment Variables

Use [`.env.example`](.env.example) as your template:

```bash
cp .env.example .env
# then edit .env with your local absolute paths and tokens
```

## Data Preparation

### 1) Seed documents into SQLite

```bash
uv run python scripts/seed_docs_sqlite.py \
  --input /path/to/wiki-abstracts-2017/collection.tsv \
  --db /path/to/wiki-abstracts-2017/docs.db \
  --delimiter $'\t' \
  --id-column id \
  --title-column title \
  --text-column text \
  --batch-size 10000
```

### 2) Build sparse index files

#### CPU (or auto-detect):

```bash
uv run python scripts/generate_sparse_index.py \
  --input /path/to/wiki-abstracts-2017/collection.tsv \
  --out-dir ./data/index \
  --batch-size 64
```

#### CUDA (multiple GPUs):

```bash
uv run python scripts/generate_sparse_index.py \
  --input /path/to/wiki-abstracts-2017/collection.tsv \
  --out-dir ./data/index \
  --target-devices cuda:0,cuda:1 \
  --batch-size 64
```

This creates:

- `term_offsets.npz`
- `doc_ids.npy`
- `term_weights.npy`

## Run the Retrieval API

```bash
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Open docs at `http://0.0.0.0:8000/docs`.

## Query Examples (bash)

### Single query (GET)

```bash
curl "http://0.0.0.0:8000/search/single?query=castle%20inherited&k=3"
```

### Single query (POST)

```bash
curl -X POST "http://0.0.0.0:8000/search/single" \
  -H "Content-Type: application/json" \
  -d '{"query":"castle inherited","k":3}'
```

### Batch query (POST)

```bash
curl -X POST "http://0.0.0.0:8000/search/batch" \
  -H "Content-Type: application/json" \
  -d '{"queries":["castle inherited","who is david gregory"],"k":3}'
```

## Example Client (`dspy.ColBERTv2`)

Start the API first, then run:

```bash
uv run python examples/splade.py
```

## Create `requirements.txt` and `requirements_dev.txt` with uv

If you need pip-style lock files (for CI, Docker, or external tooling), generate them from
`pyproject.toml`:

```bash
# runtime dependencies from [project.dependencies]
uv pip compile pyproject.toml -o requirements.txt

# dev dependency group from [dependency-groups].dev
uv pip compile --group dev -c requirements.txt -o requirements_dev.txt
```

Install from those files (inside your active virtualenv):

```bash
uv pip install -r requirements.txt
uv pip install -r requirements_dev.txt
```

## Development Commands

- Install Dev Dependencies
  ```bash
  uv sync
  ```

```bash
uv run ruff check .
uv run pre-commit run --all-files
```

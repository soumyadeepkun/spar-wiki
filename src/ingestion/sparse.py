from __future__ import annotations

import csv
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

W_MAX = 8.0
QMAX = 255
BUF_SIZE = 1_000_000


def batched(iterable, size: int):
    while True:
        batch = list(islice(iterable, size))
        if not batch:
            break
        yield batch


def process_tsv(file_path: str, model, pool, out_dir: str, bsz: int = 64):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    doc_ids = np.empty(BUF_SIZE, dtype=np.uint32)
    term_ids = np.empty(BUF_SIZE, dtype=np.uint16)
    term_weights = np.empty(BUF_SIZE, dtype=np.uint8)

    q_scale = QMAX / W_MAX

    with open(file_path, "r") as tsvfile:
        csvreader = csv.DictReader(tsvfile, delimiter="\t")
        if not csvreader.fieldnames or len(csvreader.fieldnames) < 3:
            raise ValueError("Input TSV must contain at least 3 columns: id, text, title")

        id_, text, title = csvreader.fieldnames
        i = 0
        j = 0
        k = 0

        for batch in batched(csvreader, bsz):
            ids = np.array([int(row[id_]) for row in batch], dtype=np.uint32)
            docs = [f"title:{row[title]} | text:{row[text]}" for row in batch]
            doc_embeds = model.encode_document(docs, pool=pool, batch_size=bsz)

            for idx, sp_tensor in zip(ids, doc_embeds):
                row = sp_tensor.coalesce().cpu()
                indices = row.indices().view(-1).numpy()
                vals = row.values().numpy()
                n = len(indices)

                while n > 0:
                    space = BUF_SIZE - j
                    if n <= space:
                        doc_ids[j : j + n] = idx
                        term_ids[j : j + n] = indices
                        term_weights[j : j + n] = np.floor(
                            np.clip(vals, 0.0, W_MAX) * q_scale
                        ).astype(np.uint8)
                        j += n
                        break

                    doc_ids[j:BUF_SIZE] = idx
                    term_ids[j:BUF_SIZE] = indices[:space]
                    term_weights[j:BUF_SIZE] = np.floor(
                        np.clip(vals[:space], 0.0, W_MAX) * q_scale
                    ).astype(np.uint8)

                    table = pa.table(
                        {"doc_id": doc_ids, "term_id": term_ids, "term_weight": term_weights}
                    )
                    file_name = out_path / f"splade_part_{k:04d}.parquet"
                    pq.write_table(
                        table,
                        file_name,
                        compression={
                            "doc_id": "zstd",
                            "term_weight": "zstd",
                            "term_id": "none",
                        },
                    )
                    print(f"{file_name} successfully written.")
                    k += 1

                    indices = indices[space:]
                    vals = vals[space:]
                    n -= space
                    j = 0

            i += len(batch)

        if j > 0:
            table = pa.table(
                {"doc_id": doc_ids[:j], "term_id": term_ids[:j], "term_weight": term_weights[:j]}
            )
            file_name = out_path / f"splade_part_{k:04d}.parquet"
            pq.write_table(
                table,
                file_name,
                compression={"doc_id": "zstd", "term_weight": "zstd", "term_id": "none"},
            )
            k += 1

        return i, k


def process_parquet_shards(
    in_dir: str = ".",
    in_pattern: str = "splade_part_*.parquet",
    out_dir: str = ".",
    cleanup: bool = True,
):
    start = perf_counter()

    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full_pattern = str(in_path / in_pattern)

    df_sorted = pl.scan_parquet(full_pattern).sort("term_id").collect()

    term_ids = df_sorted["term_id"].to_numpy().astype(np.uint16)
    term_changes = np.concatenate([[0], np.nonzero(np.diff(term_ids))[0] + 1, [len(term_ids)]])
    unique_term_ids = term_ids[term_changes[:-1]]
    offsets = np.column_stack([unique_term_ids, term_changes[:-1], term_changes[1:]])
    np.savez_compressed(out_path / "term_offsets.npz", offsets=offsets)
    print(f"Saved offsets for {len(unique_term_ids)} terms")

    df_sorted = df_sorted.drop("term_id")
    doc_ids = df_sorted["doc_id"].to_numpy().astype(np.uint32)
    term_weights = df_sorted["term_weight"].to_numpy().astype(np.uint8)
    np.save(out_path / "doc_ids.npy", doc_ids)
    np.save(out_path / "term_weights.npy", term_weights)

    elapsed_time = perf_counter() - start
    print(f"Elapsed Time: {elapsed_time:.6f} seconds")

    if cleanup:
        for shard in in_path.glob(in_pattern):
            shard.unlink()

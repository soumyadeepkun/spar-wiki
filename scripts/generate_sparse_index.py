from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import torch
from dotenv import load_dotenv
from sentence_transformers import SparseEncoder

from src.ingestion.sparse import process_parquet_shards, process_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sparse index from collection TSV.")
    parser.add_argument("--input", required=True, help="Path to collection.tsv")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for parquet shards and final index files",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN env var if omitted.",
    )
    parser.add_argument(
        "--model-name",
        default="naver/splade-v3",
        help="Sparse encoder model name",
    )
    parser.add_argument(
        "--target-devices",
        default=None,
        help="Optional comma-separated devices override for start_multi_process_pool",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep intermediate parquet shards",
    )
    return parser.parse_args()


def infer_target_devices() -> list[str]:
    if torch.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    return ["cpu"]


def main() -> None:
    load_dotenv(override=False)
    args = parse_args()
    input_path = str(Path(args.input).expanduser().resolve())
    out_dir = str(Path(args.out_dir).expanduser().resolve())
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF token not provided. Pass --hf-token or set HF_TOKEN.")
    if args.target_devices:
        target_devices = [x.strip() for x in args.target_devices.split(",") if x.strip()]
        if not target_devices:
            raise ValueError("--target-devices must contain at least one device.")
    else:
        target_devices = infer_target_devices()
    print(f"Using target devices: {target_devices}")

    model = SparseEncoder(args.model_name, token=hf_token)
    model.eval()
    pool = model.start_multi_process_pool(target_devices=target_devices)
    try:
        fp = input_path
        start = perf_counter()
        docs_processed, files_created = process_tsv(
            fp, model, pool, out_dir=out_dir, bsz=args.batch_size
        )
        end = perf_counter()
        elapsed_time = end - start
        print("\n\n\n")
        print(
            f"Docs Processed: {docs_processed}\n"
            f"Parquet Files Created: {files_created}\n"
            f"Elapsed Time: {elapsed_time:.6f} seconds"
        )

        process_parquet_shards(
            in_dir=out_dir,
            in_pattern="splade_part_*.parquet",
            out_dir=out_dir,
            cleanup=not args.keep_shards,
        )
    finally:
        model.stop_multi_process_pool(pool)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from pathlib import Path
from time import perf_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed an SQLite docs table from TSV/CSV.")
    parser.add_argument("--input", required=True, help="Path to input TSV/CSV file")
    parser.add_argument("--db", required=True, help="Path to output SQLite database file")
    parser.add_argument(
        "--table",
        default="docs",
        help="Destination table name (default: docs)",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Input delimiter (default: tab)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Rows per INSERT batch (default: 10000)",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Column name for document id (default: id)",
    )
    parser.add_argument(
        "--title-column",
        default="title",
        help="Column name for title (default: title)",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name for body text (default: text)",
    )
    parser.add_argument(
        "--dump-sql",
        default=None,
        help="Optional path to write sqlite .dump SQL output",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_identifier(name: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def seed_docs(
    input_path: Path,
    db_path: Path,
    table: str,
    delimiter: str,
    batch_size: int,
    id_column: str,
    title_column: str,
    text_column: str,
) -> int:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    table = validate_identifier(table)

    ensure_parent(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")

        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            f"""
            CREATE TABLE {table} (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL
            )
            """
        )

        inserted = 0
        batch: list[tuple[int, str, str]] = []
        started = perf_counter()

        with input_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            required = {id_column, title_column, text_column}
            missing = [col for col in required if col not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Missing expected columns in input: {missing}")

            for row in reader:
                doc_id = int(row[id_column])
                title = row[title_column]
                body = row[text_column]
                batch.append((doc_id, title, body))

                if len(batch) >= batch_size:
                    conn.executemany(
                        f"INSERT INTO {table} (id, title, body) VALUES (?, ?, ?)", batch
                    )
                    conn.commit()
                    inserted += len(batch)
                    batch.clear()

            if batch:
                conn.executemany(f"INSERT INTO {table} (id, title, body) VALUES (?, ?, ?)", batch)
                conn.commit()
                inserted += len(batch)

        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_id ON {table}(id)")
        conn.commit()
        elapsed = perf_counter() - started
        print(f"Inserted {inserted} rows into {db_path} in {elapsed:.2f}s")
        return inserted
    finally:
        conn.close()


def dump_sql(db_path: Path, dump_sql_path: Path) -> None:
    ensure_parent(dump_sql_path)
    conn = sqlite3.connect(str(db_path))
    try:
        with dump_sql_path.open("w", encoding="utf-8") as handle:
            for line in conn.iterdump():
                handle.write(f"{line}\n")
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    inserted = seed_docs(
        input_path=Path(args.input),
        db_path=Path(args.db),
        table=args.table,
        delimiter=args.delimiter,
        batch_size=args.batch_size,
        id_column=args.id_column,
        title_column=args.title_column,
        text_column=args.text_column,
    )

    if args.dump_sql:
        dump_sql(db_path=Path(args.db), dump_sql_path=Path(args.dump_sql))
        print(f"Wrote SQL dump to {args.dump_sql}")

    print(f"Done. Total rows seeded: {inserted}")


if __name__ == "__main__":
    main()

"""
List PDF sources in the datastore: child chunk count and how many are embedded.

  python -m phase3_rag.sources
"""
from __future__ import annotations

import argparse
import sys

from common.datastore_config import open_chunk_store


def format_rows(rows: list[tuple[str, int, int]]) -> str:
    if not rows:
        return "(no pdf sources)"
    name_w = max(9, max(len(r[0]) for r in rows))
    header = f"{'source_id':<{name_w}}  {'chunks':>8}  {'embedded':>8}"
    lines = [header, "-" * len(header)]
    for source_id, chunks, embedded in rows:
        lines.append(f"{source_id:<{name_w}}  {chunks:8d}  {embedded:8d}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="List PDF chunk/embedding counts from Postgres")
    ap.add_argument("--datastore-config", default="config/datastore.yaml")
    a = ap.parse_args(argv)
    store = open_chunk_store(a.datastore_config)
    print(format_rows(store.list_pdf_sources()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

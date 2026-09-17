#!/usr/bin/env python3
"""
Build the Tier B concept index from course_data/concepts.csv.

WHERE THE LOGIC LIVES NOW

    Chunking, embedding and persistence moved to vat-research/vector_index/.
    The lint, the per-module coverage inventory, and the rule that one row is
    one chunk that is never split live in vector_index/adapters/concepts.py.

    The normal way to rebuild is the vector-index MCP's build_concept_index
    tool. This script is the break-glass path.

    NOT on the after-class cadence: rebuild this when concepts are written or
    edited, which is a different rhythm from the Canvas sync.

Usage:
    python scripts/build_concepts.py --dry-run   # report + lint, embed nothing
    python scripts/build_concepts.py             # write the index
    python scripts/build_concepts.py --strict    # exit non-zero on any lint error
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from scripts._vector_index_shim import run_build  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build the Tier B concept index.")
    ap.add_argument("--course", default="isom550")
    ap.add_argument("--dry-run", action="store_true", help="report + lint, embed nothing")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any lint error")
    args = ap.parse_args()
    run_build(args.course, "concepts", dry_run=args.dry_run, strict=args.strict)


if __name__ == "__main__":
    main()

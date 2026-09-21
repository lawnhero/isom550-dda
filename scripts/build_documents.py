#!/usr/bin/env python3
"""
Build the class-document index: class recaps and assignment briefs.

(Tier C in the architecture docs. Tier names describe positions in the
retrieval stack and stay in prose; identifiers say what the thing IS.)

WHERE THE LOGIC LIVES NOW

    Chunking, embedding and persistence moved to vat-research/vector_index/
    when a second course needed the same pipeline. The engine there is generic;
    what a class document IS -- announcement bodies and assignment
    instructions, split only past ~1800 characters with the title re-attached
    to every piece -- lives in vector_index/adapters/documents.py.

    The normal way to rebuild is the vector-index MCP's build_document_index
    tool, which also writes the provenance stamp that lets the app notice when
    this index has fallen behind course_data/schedule.json. This script is the
    break-glass path.

Usage:
    python scripts/build_documents.py --dry-run   # chunk report, embeds nothing
    python scripts/build_documents.py             # writes data/documents

    # if vat-research lives somewhere else:
    export CANVAS_SYNC_HOME=/path/to/vat-research

Sync first -- this index is only ever as current as the snapshot it reads:
    (via the canvas-course-sync skill, or scripts/sync_canvas.py)
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from scripts._vector_index_shim import run_build  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build the class-document index.")
    ap.add_argument("--course", default="isom550")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be indexed, embed nothing")
    args = ap.parse_args()
    run_build(args.course, "documents", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
